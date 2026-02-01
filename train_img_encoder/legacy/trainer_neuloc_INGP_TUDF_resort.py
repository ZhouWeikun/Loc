# -*- coding: utf-8 -*-
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from __future__ import print_function, division
import argparse
import torch
import tqdm
# from torch.ao.nn.quantized.functional import threshold
from torch.cuda.amp import GradScaler
# import torch.nn.functional as F
import time
import scipy.io
# from triton.language import dtype

# from tool.util_mk_optimizer import  make_optimizer
from train_img_encoder.nets_taskflow import make_img_encoder
from tool.utils_fm_duav import get_logger, get_unique_exp_dir
# from tool.utils import load_network_wstate, save_network_wstate
import warnings
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import torchvision
# import torch.nn.functional as F
import glob
import math

# from losses.loss_cl import Loss
warnings.filterwarnings("ignore")

# var to selct:
# from datasets.make_dataloader import make_dataloader
# from datasets_custom.make_dataloder_classify import make_dataloader_train
# from datasets_custom.make_dataloader_xmu import make_dataloader_xmu
# from datasets_custom.make_dataloader_gta import make_dataloader_gta
# from exps.exp24.datasets_custom.make_dataloader_dsalad import  make_dataloader
import yaml
import os
import json

from tool.util_mk_optimizer import create_optimizer_w_temple
from tool.util_ckpt_handler import load_param
from models.pos_encoder import encode_4d_coords
from util_unravel_index import unravel_index

def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    # about exp setting
    parser.add_argument('--exps_dir', default='.exps/', type=str, help='the dir that save experiments')
    parser.add_argument('--exp_name', default='debug', type=str,
                        help='the experiment name that will be saved in exps dir in the root')
    parser.add_argument('--p_yaml', default='/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/opts_wingtra.yaml',
                        type=str, help='the yaml file about the defult setting')
    parser.add_argument('--p_satinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu__512h_lineClassed/dataset_xmu_meta/satimg_xiangan_xmu_info.json',
                        type=str, help='')
    parser.add_argument('--p_uavinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu_512h_lineClassed/dataset_xmu_meta/uavimgs_xiangan_xmu_info.json',
                        type=str, help='')
    parser.add_argument('--p_uav_geocsv',
                        default='/home/data/zwk/data_uavimgs_wingtra/Zurich/IMAGES_info/uavimgs_geo_corrected_v1.csv',
                        type=str, help='')
    parser.add_argument('--dataset_name', default='xmu', type=str)
    parser.add_argument('--load2test', default="/home/data/zwk/pyproj_neuloc_v0/exps/exp_wohead_vit-b/epoch000.pth",
                        type=str, help='path for testing')  # for testing
    parser.add_argument('--load2train', default="", type=str,
                        help='exps path for pre-loading')  # for continuing training
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--val_freq', default=10, type=int)
    parser.add_argument('--tensorboard', action='store_true', default=True)
    parser.add_argument('--n_satrand_per_uav', default=8, type=int, help='will be used in dataset')
    # about hardware
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--num_worker', default=16, type=int, help='')
    parser.add_argument('--batchsize_sat', default=32, type=int, help='batchsize')
    parser.add_argument('--batchsize_uav', default=32, type=int, help='batchsize')
    parser.add_argument('--autocast', action='store_true', default=True, help='use mix precision')
    # about data setting,version 2:
    parser.add_argument('--imgsize2net', default=224, type=int)
    parser.add_argument('--satimgsize2crop', default=224, type=int)
    # parser.add_argument('--n_rand2sample_per_pos', default=2, type=int)
    # parser.add_argument('--uav_da', nargs='+', default=['rr'],help='rr=random_rotate,ra=random affine,re=random erasing,cj=color jitter,cda=color data argument')
    # parser.add_argument('--sat_da', nargs='+', default=['ra','re'],help='rr=random_rotate,ra=random affine,re=random erasing,cj=color jitter,cda=color data argument')
    # parser.add_argument('--erasing_p', default=0.3, type=float,help='random erasing probability, in [0,1]')
    # about networks
    parser.add_argument('--backbone', default="ViTB-384", type=str, help='ViTB-224;ViTS-224;dinov2_vitb14;ViTB-384')
    parser.add_argument('--head', default="", type=str, help='salad;FSRA;LPN;')  # "" means no head
    parser.add_argument('--block', default=2, type=int, help='')  # will by used when headF=FSRA,LPN,NetVLAD,NeXtVLAD
    parser.add_argument('--num_bottleneck', default=512, type=int, help='the dimensions for embedding the feature')
    parser.add_argument('--head_pool', default="avg", type=str,
                        help='head pooling type for applying')  # will by used when head=SingleBranch
    parser.add_argument('--wcls_token', default=False, type=bool)  # will by used when head=SingleBranch
    parser.add_argument('--norm_output', default=True, type=bool)
    parser.add_argument('--w_classify', default=False, action='store_true', help='')
    parser.add_argument('--feature_loss', nargs='+', default=["WeightedSoftTripletLoss"],
                        help='"InfoNceLoss","MSLoss","TripletLoss","HardMiningTripletLoss","SameDomainTripletLoss","WeightedSoftTripletLoss","ContrastiveLoss"')
    parser.add_argument('--cls_loss', default="CELoss", type=str, help='loss type of representation learning')
    parser.add_argument('--kl_loss', default="KLLoss", type=str, help='loss type of mutual learning')

    # about learning setting
    parser.add_argument('--num_epochs', default=50, type=int, help='total epoches for training')
    parser.add_argument('--warm_epoch', default=0, type=int,
                        help='the first K epoch that needs warm up')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')

    def json_dict(value):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise argparse.ArgumentTypeError("Invalid JSON format for dictionary.")

    parser.add_argument(
        "--optimizer",
        type=json_dict,
        default='{"name":"sgd","lr":0.01,"weight_decay":5e-4,"momentum":0.9,"nesterov":true}',
        help="Dictionary parameter in JSON format."
    )
    parser.add_argument(
        "--lr_sched",
        type=json_dict,
        default='{"name":"multistep","milestones": [35,45],"gamma":0.95 }',
        help="Dictionary parameter in JSON format."
    )

    # 优先级：命令行参数 > YAML 文件参数 > Python 脚本中的默认参数
    # --- 获取命令行参数的原始默认值 ---
    # parse_args([]) 或 parse_args(args=[]) 会解析空列表作为命令行参数，从而获取所有参数的默认值, 这避免了命令行参数被 YAML 覆盖
    default_args = parser.parse_args(args=[])

    # --- 解析命令行参数 (真实传入的参数) ---
    # 这一步将解析用户在命令行中实际传入的参数，如果某个参数在命令行中被指定，它将覆盖 default_args 中的值。
    opt = parser.parse_args()

    # --- 读取 YAML 文件并更新默认值 ---
    yaml_file_path = opt.p_yaml  # 获取 YAML 文件路径
    if os.path.exists(yaml_file_path):
        print(f"从 YAML 文件 '{yaml_file_path}' 加载配置...")
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            # 遍历 YAML 配置并更新 opt 对象
        for section, params in yaml_config.items():
            if isinstance(params, dict):
                # 处理嵌套的配置 (例如 Data_Augmentation_Setting)
                for key, value in params.items():
                    # 只有当 opt 中对应的属性是其默认值时，才用 YAML 的值覆盖，即优先级：命令行参数>yaml文件参数>argparse默认参数
                    # 这样确保了命令行参数优先
                    if hasattr(opt, key) and getattr(opt, key) == getattr(default_args, key):
                        # 对于 nargs='+' 的参数，如 uav_da, sat_da，yaml 读取进来是 list
                        # argparse 也会解析成 list，直接赋值即可
                        setattr(opt, key, value)
            else:
                # 处理非嵌套的顶级配置
                if hasattr(opt, section) and getattr(opt, section) == getattr(default_args, section):
                    setattr(opt, section, params)

    # --- 组织参数到 group_dict,为了后续保存为yaml时按分层组织 ---
    group_info = {
        'exp_setting': ['p_yaml', 'p_satinfo_json', 'p_uavinfo_json', 'p_uav_geocsv', 'exp_name', 'exps_dir',
                        'load2train', 'load2test', 'val', 'val_freq', 'save_freq', 'tensorboard'],
        'hardware_setting': ['gpu_ids', 'num_worker', 'batchsize_sat', 'batchsize_uav', 'autocast'],
        'data_setting': ['imgsize2net', 'satimgsize2crop', ],
        'learning_setting': ['warm_epoch', 'num_epochs', 'droprate', 'optimizer', 'lr_sched'],
        'network_setting': ['w_classify', 'block', 'cls_loss', 'feature_loss', 'kl_loss', 'num_bottleneck', 'backbone',
                            'head', 'head_pool', 'wcls_token', 'norm_output']  # 补上 wcls_token 和 norm_output
    }
    opt.group_dict = group_info
    print(opt)  # 打印最终的参数

    return opt


class UDFComputer(object):
    def __init__(self, sat_dataset):
        self.sat_dataset = sat_dataset

        # 定义距离归一化因子 (根据实验调整)
        self.norm_factor_rc = math.sqrt(self.sat_dataset.nr2sample_h ** 2 + self.sat_dataset.nc2sample_w ** 2)
        self.nrom_factor_rot = torch.pi  # todo:make the threshold auto
        self.nrom_factor_scale = math.log(self.sat_dataset.satimgsize_scale_to_200m_boundary[1] / self.sat_dataset.satimgsize_scale_to_200m_boundary[0])

        # weight definity,version1:
        self.w_rc = 0.6  # 位置权重，通常设为1.0作为基准
        self.w_r = 0.3  # 位置权重，通常设为1.0作为基准
        self.w_s = 0.1  # 尺度权重
        rc_dist_threshold_accpetable = self.sat_dataset.halfimg_radius_nrc
        # self.weight_rc_dist_func = lambda x:torch.sigmoid(10*x/rc_dist_threshold_accpetable-5) #weight in [0,1],assuming x/rc_dist_threshold_accpetable mapping rc_dist_threshold_accpetable to 1
        # self.neg_weight_fm_nrc_dist = lambda x:torch.sigmoid(7.5*x/rc_dist_threshold_accpetable-3.75) #x/rc_dist_threshold_accpetable -> mapping rc_dist_threshold_accpetable to 1
        self.neg_weight_fm_nrc_dist = lambda x: torch.sigmoid(8.5 * x / rc_dist_threshold_accpetable - 4.68)  # x/rc_dist_threshold_accpetable -> mapping rc_dist_threshold_accpetable to 1
        # self.neg_weight_fm_nrc_dist = lambda x: torch.sigmoid(8.5 * (x / rc_dist_threshold_accpetable - 0.25))
        self.udf_threshold_accpetable = rc_dist_threshold_accpetable

    def compute_udf_fm_diff(self, dists_rc, dists_rot, dists_scale=None):
        dists_rc_normed = dists_rc / self.norm_factor_rc
        dists_rot_normed = dists_rot / self.nrom_factor_rot
        dists_scale_normed = dists_scale / self.nrom_factor_scale if dists_scale is not None else None

        # 计算加权的平方和, version2,
        rc_err = dists_rc_normed
        neg_weight = self.neg_weight_fm_nrc_dist(dists_rc_normed)
        rot_err = dists_rot_normed + (1 - dists_rot_normed) * neg_weight
        rot_term = torch.clamp(rot_err, max=1.0)  # min(1.,scale_err)
        if dists_scale is not None:
            scale_err = dists_scale_normed + (1 - dists_scale_normed) * neg_weight
            scale_term = torch.clamp(scale_err, max=1.0)  # min(1.,scale_err)
            dist_total_sq = self.w_rc * rc_err ** 2 + self.w_r * rot_term ** 2 + self.w_s * scale_term ** 2
        else:
            dist_total_sq = self.w_rc * rc_err ** 2 + self.w_r * rot_term ** 2

        #  取平方根，得到最终的距离,这使得 dist_true 的“单位”与 dist_pred 保持一致，损失函数更稳定
        dist_label = torch.sqrt(dist_total_sq) + 1e-7
        return dist_label


class Trainer(object):
    def __init__(self):
        self.opt = get_parse()

        if torch.cuda.is_available():
            device = torch.device("cuda:" + self.opt.gpu_ids[0])
            self.opt.use_gpu = True
        else:
            device = torch.device("cpu")
            self.opt.use_gpu = False
        self.device = device

        # 1.config the img_encoder
        self.img_encoder = make_img_encoder(self.opt).to(self.device)
        feat_q_dim = self.img_encoder.backbone.output_channel
        self.feat_q_dim = feat_q_dim
        # freeze the para
        # for param in self.img_encoder.parameters():
        #     param.requires_grad = False

        # 2.config the aggregator
        # self.decoder = LocalDecoderFiLM(dim=feat_q_dim,c_dim=feat_q_dim,hidden_size=1024,n_blocks=3,output_dim=1,norm_type='none',leaky=True).to(self.device)
        # from models.Head.G2M import G2M
        # self.img_aggregator = G2M(in_channels=feat_q_dim,out_channels=feat_q_dim,rank=1024).to(self.device)
        # from models.Head.salad import SALAD
        # self.img_aggregator = SALAD(input_feat_dim=feat_q_dim, global_token_dim=128, pathchsize=16, num_clusters=14, cluster_dim=64).to(self.device)
        from models.Head.salad_residual import SALAD_Residual
        self.img_aggregator = SALAD_Residual(input_feat_dim=feat_q_dim, base_dim=feat_q_dim, patchsize=16,
                                             num_clusters=16, cluster_dim=64).to(self.device)
        # from models.Head.salad_film import SALAD_FiLM
        # self.img_aggregator = SALAD_FiLM(input_feat_dim=feat_q_dim,base_dim=feat_q_dim,patchsize=16, num_clusters=16, cluster_dim=64).to(self.device)
        self.agg_name = 'salad'
        # load the param for trained aggregator:
        p2load = '/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/exps/zurich_12km2_cl_saladResidual_ep3.6k_swllossWweight/epoch3969.pth'
        load_param(p2load, {'aggregator': self.img_aggregator})

        # 3.config the pos_encoder
        from models.pos_encoder import PositionalEncoder
        self.rc_pos_encoder = PositionalEncoder(input_dims=2, include_input=True, multires=8)
        self.rot_pos_encoder = PositionalEncoder(input_dims=2, include_input=True, multires=6)
        self.scale_pos_encoder = PositionalEncoder(input_dims=1, include_input=True, multires=4)
        self.coord_encoded_dim = self.rc_pos_encoder.out_dim + self.rot_pos_encoder.out_dim + self.scale_pos_encoder.out_dim

        # 4.config the grid
        from app.nerf.main_nerf import NeRFAppConfig
        from wisp.config._tyro import parse_args_tyro_v1
        self.grid_args = parse_args_tyro_v1(NeRFAppConfig,
                                            '/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/nerf_hash.yaml')
        from wisp.config import instantiate
        blas = instantiate(self.grid_args.blas, pointcloud=None)
        self.grid = instantiate(self.grid_args.grid, blas=blas).to(self.device)  # A grid keeps track of both features and occupancy
        # from models.multi_mlp import create_mlp
        # self.grid_mlp = create_mlp([coord_dim+feat_q_dim,feat_q_dim,feat_q_dim],norm_type='layer').to(self.device)
        # from models.ocn_mlp import LocalDecoder,LocalDecoderFiLM
        # self.grid_mlp = SerialModulator(s_dim=feat_q_dim,c_dim=coord_dim+feat_q_dim, hidden_size=1024,n_blocks=5,output_dim=1024,c_operation='add',leaky=True).to(self.device)
        from models.cond_modulator_shallow_serial import SerialModulatorShallow
        self.grid_mlp = SerialModulatorShallow(input_dim=feat_q_dim, condition_dim=self.coord_encoded_dim,
                                               hidden_dim=512, num_blocks=1, output_dim=feat_q_dim,
                                               condition_operator='add',
                                               ).to(self.device)
        p2load = '/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/exps/zurich_12mk2_hashfit_ep5k/epoch5000.pth'
        load_param(p2load, {
            'grid': self.grid,
            'grid_mlp': self.grid_mlp,
        })

        # 5.config the metricnet(tudf_predictor)
        from models.rank_former import RankFormer,QueryRefProjection
        self.rank_former = RankFormer(
            input_dim=512,
            tf_dim_feedforward=1024,
            tf_nhead=1,
            tf_num_layers=2,
            head_hidden_layers=None,
            dropout=0.2,
            fixed_length=True
        ).to(self.device)
        self.qf_proj = QueryRefProjection(
            feat_dim=self.feat_q_dim,
            coord_dim=self.coord_encoded_dim,
            output_dim=512,
            use_layer_norm=False,
            use_activation=False,
            activation='gelu'
        ).to(self.device)

        # define the param to save/laod
        self.param2optimize = {
            # 'img_aggregator':self.img_aggregator,
            # 'grid':self.grid,
            # 'grid_mlp':self.grid_mlp,
            # 'metric_net': self.metric_net,
            "rank_former": self.rank_former,
            "qf_proj": self.qf_proj,
        }
        self.param2freeze = {
            'img_encoder': self.img_encoder,
            'aggregator': self.img_aggregator,
            'grid': self.grid,
            'grid_mlp': self.grid_mlp,
        }
        for name, module in self.param2freeze.items():
            for param in module.parameters():
                param.requires_grad = False

    def _test_ready(self):
        opt = self.opt
        # load the training config to update opt
        suffix = '.yaml'
        pattern = os.path.join(os.path.dirname(opt.load2test), f'*{suffix}')
        config_path = glob.glob(pattern)[0]
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        for group_dict_key, group_dict in config.items():
            if group_dict_key == 'network_setting':
                for cfg, value in group_dict.items():
                    setattr(opt, cfg, value)
            else:
                for cfg, value in group_dict.items():
                    if not hasattr(opt, cfg):
                        setattr(opt, cfg, value)

        # laod the ckpt
        from tool.util_ckpt_handler import load_param
        load_param(opt.load2test, self.param2optimize)
        for k, v in self.param2optimize.items():
            v.eval()
        for k, v in self.param2freeze.items():
            v.eval()

        # config the datalaoder
        from dataset_wingtra_4d import SatDataset
        self.sat_dataset = SatDataset(
            p_satinfo_json=self.opt.p_satinfo_json,
            p_uav_geocsv=self.opt.p_uav_geocsv,
            imgsize2net=224,
        )
        self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset, batch_size=1,
                                                          num_workers=self.opt.num_worker,
                                                          pin_memory=True, shuffle=True, drop_last=True,
                                                          persistent_workers=True)

    def train(self):
        opt = self.opt

        # config the optimizer
        self.optimizer = create_optimizer_w_temple(self.param2optimize, 'adam')
        self.scaler = GradScaler()

        # load the ckpt for continuing train if necessray
        from tool.util_ckpt_handler import load_param
        if opt.load2train is not None and len(opt.load2train) > 0:
            params_to_add = {"optimizer_state": self.optimizer}
            self.param2optimize.update(params_to_add)
            load_param(opt.load2train, self.param2optimize)
        begin_epoch = 0

        # config the datalaoder:
        from dataset_wingtra_4d import SatDataset
        self.sat_dataset = SatDataset(
            p_satinfo_json=self.opt.p_satinfo_json,
            p_uav_geocsv=self.opt.p_uav_geocsv,
            imgsize2net=224,
        )
        self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset, batch_size=self.opt.batchsize_sat,
                                                          num_workers=self.opt.num_worker,
                                                          pin_memory=True, shuffle=True, drop_last=True,
                                                          persistent_workers=True)
        # config the ref_feat_gallery
        # self.gallery_dict = self._get_feat_gallery_fm_grid(overlap=0.25,scale=self.sat_dataset.satimg
        # size_scale_to_200m_unif_list[2])
        self.feat_gallery_dict = self._get_feat_gallery_fm_grid(overlap=0.25,scale=None)
        import faiss
        self.feat_gallery_index = faiss.IndexHNSWFlat(self.feat_q_dim, 36)  # 使用简单的暴力L2索引作为例子
        self.feat_gallery_index.add(self.feat_gallery_dict['feat_gallery_flatten'].detach().cpu().numpy())
        self.feat_gallery_index.hnsw.efSearch = 64  # 值越高，越准，越慢。)

        # config the logger&writer
            # make the dir to save the exp
        exp_name = get_unique_exp_dir(opt.exps_dir, opt.exp_name)
        opt.exp_name = exp_name
        exp_dir2save = os.path.join(opt.exps_dir, opt.exp_name)
        os.makedirs(exp_dir2save, exist_ok=True)
            # config the logger
        logger = get_logger("{}/{}/train.log".format(opt.exps_dir, opt.exp_name), 'trainer_logger')
        self.logger = logger
        self.logger.info(f"exp ready!, exp_name={exp_name}")
            # config tensorborad if necessary
        self.writer = SummaryWriter("exps/{}/train_tensorboard.log".format(opt.exp_name)) if opt.tensorboard else None

        # config about loss
        self.udf_compter = UDFComputer(sat_dataset=self.sat_dataset)
        loss_mse = torch.nn.MSELoss(reduction='mean')
        from losses.WeightedSoftTripletLoss_fm_mat import SWTLoss_fm_mat, MSLoss_fm_mat
        loss_swt = SWTLoss_fm_mat(decoupling=False, )
        loss_ms = MSLoss_fm_mat()
        temperature = 1.
        targets = torch.arange(opt.batchsize_sat).to(self.device)  # [0, 1, 2, ..., B-1]

        # ready to trian
        since = time.time()
        step = 0
        num_epochs = opt.num_epochs
        for epoch in range(begin_epoch, num_epochs):
            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            for it, data in tqdm.tqdm(enumerate(self.sat_dataloader)):
                satimg, sat_nrc, rad_roted, satimgsize_cover_ratio = data
                # uav_imgs, uav_nrcs = next(iter(self.uav_dataloader_train))

                gt_coords = torch.concatenate([sat_nrc, rad_roted, satimgsize_cover_ratio], 1).to(self.device)
                # coords = torch.concatenate([gt_coords.unsqueeze(1), gt_coords.unsqueeze(1)], dim=1)
                # coords_flatten = coords.flatten(start_dim=0,end_dim=1)
                # satimgs_q = satimg.flatten(start_dim=0,end_dim=1)
                # satimg_q_ids = [coords.shape[1] * i for i in range(coords.shape[0])]

                satimgs_q = satimg
                with torch.no_grad():
                    # get feats from visual_encoder
                    feats_q = self.img_encoder(satimgs_q.to(self.device))
                    if self.agg_name == 'g2m':
                        feats_q = self.img_aggregator(
                            feats_q[:, 1:, :].permute(0, 2, 1).reshape(satimgs_q.shape[0], self.feat_q_dim, 14, 14))  # for patchsize=16,imgszie=224
                    elif self.agg_name == 'salad':
                        feats_q = self.img_aggregator(feats_q)  # aggregator = salad

                    # get feats from hash
                    # feats_grid = self._feats_fm_grid(gt_coords)
                    # coords_encoded = encode_4d_coords(
                    #     torch.concatenate([sat_nrc, rad_roted, satimgsize_cover_ratio], dim=-1),
                    #     rc_encoder=self.rc_pos_encoder,
                    #     rot_endcoder=self.rot_pos_encoder,
                    #     scale_encoder=self.scale_pos_encoder)
                    # feats_grid = self.grid_mlp(inputs=feats_grid,
                    #                            condition_features=coords_encoded.to(feats_grid.device))
                    # feats_grid = torch.nn.functional.normalize(feats_grid, dim=-1)

                    #get pred&gt
                    distances, indices = self.feat_gallery_index.search(feats_q.detach().cpu().numpy(), k=20)
                    # condition_features=torch.concatenate([feats_grid, coords_encoded.to(feats_grid.device)],dim=-1))

                    # compute dist_mat form coords
                    coords_galleyr_flatten = self.feat_gallery_dict['coords_gallery_flatten'].to(gt_coords.device)
                    gallery_shape = self.feat_gallery_dict['gallery_shape']
                    rc_dist_mat = gt_coords[:, :2].unsqueeze(1) - coords_galleyr_flatten[:, :2].unsqueeze(0)
                    rc_dist_mat = torch.norm(rc_dist_mat, dim=-1)
                    r_dist_mat = gt_coords[:, 2].unsqueeze(1) - coords_galleyr_flatten[:, 2].unsqueeze(0)
                    r_dist_mat = torch.abs(torch.atan2(torch.sin(r_dist_mat), torch.cos(r_dist_mat))).squeeze()  # atan2 函数的输出范围是 [-π, π]
                    s_dist_mat = gt_coords[:, 3].unsqueeze(1) / coords_galleyr_flatten[:, 3].unsqueeze(0)
                    s_dist_mat = torch.abs(torch.log(s_dist_mat)).squeeze()  # 比例关系在对数空间中会变为加减关系
                    udf_dist_mat = self.udf_compter.compute_udf_fm_diff(rc_dist_mat, r_dist_mat, s_dist_mat)
                    positive_mat = udf_dist_mat < self.sat_dataset.halfimg_radius_nrc

                    #debug:
                    flatten_id = torch.argmin(rc_dist_mat,dim=-1)
                    rc_dist_mat_shaped = rc_dist_mat.reshape(-1,*gallery_shape)
                    shaped_id = unravel_index(flatten_id,gallery_shape)
                    r_dist_mat_shaped = r_dist_mat.reshape(-1,*gallery_shape)
                    r_dist_mat_shaped[0,shaped_id[0][0],shaped_id[1][0]]


                # construct the metirc_net's output
                B = feats_q.shape[0]
                # chunk_size=64
                # device = feats_grid.device
                # metric_dist_list = []
                # for query_chunk in torch.split(feats_q, chunk_size, dim=0):
                #     chunk_size_actual = query_chunk.shape[0]  # 最后一块可能不足 chunk_size
                #     # 为这一块 query 构造配对
                #     query_expanded = query_chunk.unsqueeze(1).expand(chunk_size_actual, B, -1)  # [chunk, B, D]
                #     ref_expanded = feats_grid.unsqueeze(0).expand(chunk_size_actual, B, -1)  # [chunk, B, D]
                #     coords_encoded_expanded = coords_encoded.unsqueeze(0).expand(chunk_size_actual, B, -1).to(device)  # [chunk, B, 4]
                #     # 计算这一块的距离
                #     distances_chunk = self.metric_net(
                #         query_expanded,
                #         ref_expanded,
                #         coords_encoded_expanded
                #     ).reshape(chunk_size_actual, B)  # [chunk, B]
                #     metric_dist_list.append(distances_chunk)
                # metric_dist_mat = torch.cat(metric_dist_list, dim=0)  # [B, B]
                # 一次性计算所有距离
                query_expanded = feats_q.unsqueeze(1).expand(B, B, -1)  # [B, B, D]
                ref_expanded = feats_grid.unsqueeze(0).expand(B, B, -1)  # [B, B, D]
                coords_encoded_expanded = coords_encoded.unsqueeze(0).expand(B, B, -1).to(
                    ref_expanded.device)  # [B, B, coord_dim]
                metric_dist_mat = self.metric_net(
                    query_expanded,
                    ref_expanded,
                    coords_encoded_expanded
                )  # 直接输出 [B, B]
                metric_dist_mat = torch.nn.functional.softplus(metric_dist_mat)
                # metric_dist_mat_np = metric_dist_mat.detach().cpu().numpy()

                # compute dist_mat form coords
                rc_dist_mat = gt_coords[:, :2].unsqueeze(1) - gt_coords[:, :2].unsqueeze(0)
                rc_dist_mat = torch.norm(rc_dist_mat, dim=-1)
                r_dist_mat = gt_coords[:, 2].unsqueeze(1) - gt_coords[:, 2].unsqueeze(0)
                r_dist_mat = torch.abs(
                    torch.atan2(torch.sin(r_dist_mat), torch.cos(r_dist_mat))).squeeze()  # atan2 函数的输出范围是 [-π, π]
                s_dist_mat = gt_coords[:, 3].unsqueeze(1) / gt_coords[:, 3].unsqueeze(0)
                s_dist_mat = torch.abs(torch.log(s_dist_mat)).squeeze()  # 比例关系在对数空间中会变为加减关系
                udf_dist_mat = self.udf_compter.compute_udf_fm_diff(rc_dist_mat, r_dist_mat, s_dist_mat)
                positive_mat = udf_dist_mat < self.sat_dataset.halfimg_radius_nrc
                # udf_dist_mat_np = udf_dist_mat.detach().cpu().numpy()

                # loss =  loss_swt(metric_dist_mat,positive_mat,udf_dist_mat,w_weight=True)
                # loss =  loss_ms(metric_dist_mat,positive_mat,udf_dist_mat)
                logits = -metric_dist_mat / temperature  # [B, B]
                loss = torch.nn.functional.cross_entropy(logits, targets)
                metric_dist_mat_np = metric_dist_mat.detach().cpu().numpy()

                # from util_gen_coord_samples_hierarchical import generate_pose_samples_hierarchical,get_stratified_sampling_configs,visualize_hierarchical_samples
                # config_sampling = get_stratified_sampling_configs(
                #     base_rc_std = self.sat_dataset.halfimg_radius_nrc,
                #     base_dir_std_rad = np.deg2rad(10),
                #     base_log_s_std = 0.05,
                # )
                # n_uniform_samples=0
                # coords_sampled = generate_pose_samples_hierarchical(
                #     p_true_batch=gt_coords,
                #     rc_bounds=(self.sat_dataset.nr2sample_range, self.sat_dataset.nc2sample_range),
                #     scale_bounds=self.sat_dataset.satimgsize_scale_to_200m_boundary,
                #     sampling_configs=config_sampling, # <--- 传入动态生成的配置
                #     num_uniform_samples=n_uniform_samples,
                # ).to(self.device)
                # debug:vis the coord_samples
                # visualize_hierarchical_samples(
                #     p_true=gt_coords[0],
                #     rc_bounds=(self.sat_dataset.nr2sample_range, self.sat_dataset.nc2sample_range),
                #     sampling_configs=config_sampling,
                #     all_samples=coords_sampled[0],
                #     num_uniform_samples=n_uniform_samples,
                # )
                # coords_mixed = torch.concatenate([gt_coords.unsqueeze(1),coords_sampled],dim=1)

                # 反向传播
                self.optimizer.zero_grad()
                if opt.autocast:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # 输出loss指标
                if it % 10 == 0:
                    if self.writer is not None:
                        self.writer.add_scalar('loss_it', loss, step)
                step = step + 1

            # modify the learning rate
            # scheduler.step()

            # save the network's para
            if (epoch % 10 == 0) and (epoch > 0):  # for running
                # if epoch % 2 == 0: #for debugging
                # if (epoch == 10) or (epoch % 10 == 9 and epoch >= 110):
                # if ((epoch > 0) and (epoch % opt.save_freq == 0)) or ( epoch % 10 == 9 and epoch >= 110 ):
                from tool.util_ckpt_handler import save_param
                params_to_add = {
                    "optimizer_state": self.optimizer,
                    "epoch": epoch,
                }
                self.param2optimize.update(params_to_add)
                save_param(opt.exp_name, self.param2optimize)
                # log info:
                # grid_feats_grad = self.grid.codebook.feats.grad
                # self.logger.info(f"Grid feats grad L2 norm:{torch.linalg.norm(grid_feats_grad).item()}")
                # self.logger.info(f"Grid feats value.max():{feats_grid.max().item():.4f}")
                # mean_dino = torch.mean(feats_q, dim=0)
                # mean_grid = torch.mean(feats_grid, dim=0)
                # l2_distance = torch.linalg.norm(mean_dino - mean_grid).item()
                # cosine_sim = torch.nn.functional.cosine_similarity(mean_dino.unsqueeze(0), mean_grid.unsqueeze(0)).item()
                # self.logger.info(f"  mean值- L2 距离: {l2_distance:.6f}")
                # self.logger.info(f"  mean值- 余弦相似度: {cosine_sim:.6f}")
                # var_per_dim_grid = torch.var(feats_grid, dim=0)
                # var_per_dim_dino = torch.var(feats_q, dim=0)
                # self.logger.info(f"gird_方差均值: {var_per_dim_grid.mean().item():.6f}")
                # self.logger.info(f"feat_方差的均值: {var_per_dim_dino.mean().item():.6f}")
                # # 更有意义的指标：方差的方差（衡量特征维度的差异性）
                # var_of_var = torch.var(var_per_dim_dino)
                # self.logger.info(f"feat_方差的方差: {var_of_var.item():.6f}")

                # if self.writer is not None:
                #     self.writer.add_scalar(' mean_L2_dist', l2_distance, epoch)
                #     self.writer.add_scalar(' mean_cosine_sim', cosine_sim, epoch)
                #     self.writer.add_scalar(' gird_var_mean', var_per_dim_grid.mean().item(), epoch)
                # self.writer.add_scalar(' feat_mean', mean_dino.mean().item(), epoch)
                # self.writer.add_scalar(' feat_var_mean', var_per_dim_dino.mean().item(), epoch)
                # self.writer.add_scalar(' feat_var_var', var_of_var.item(), epoch)

            # log info
            str2log = ''
            str2log += f'epcoh={epoch}'
            logger.info(f'loss={loss.item()}')
            time_elapsed = time.time() - since
            since = time.time()
            logger.info('epoch{:.0f} finished in {:.0f}m {:.0f}s'.format(epoch, time_elapsed // 60, time_elapsed % 6))
            logger.info('-' * 50)
            if self.writer is not None:
                self.writer.add_scalar('loss_epoch', loss, epoch)

            # backup the py info, at least have trained a epoch
            if epoch == 0:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(exp_dir2save, self.opt)

    def _feats_fm_grid(self, coords):
        if len(coords.shape) == 3:
            n, m, _ = coords.shape
            coords_flatten = coords.flatten(start_dim=0, end_dim=1)
        else:
            coords_flatten = coords
        scales = coords_flatten[..., -1]

        grid_nrc_coords = coords_flatten[:, :2] * 2 - 1.
        grid_rot_coords = coords_flatten[:, 2:3] / (2 * torch.pi) - 1.
        grid_rot_coords *= (180 / self.grid_args.grid.max_grid_res)
        gird_3d_coords = torch.concatenate([grid_nrc_coords, grid_rot_coords], dim=-1)
        n_gird_lod = len(self.grid.active_lods)
        feats_grid = self.grid.interpolate(gird_3d_coords.to(self.device), n_gird_lod - 1)
        #   aggerate the multiscale feats
        # normalized_scales = (scales - self.sat_dataset.satimgsize_scale_to_200m_boundary[0]) / (
        #             self.sat_dataset.satimgsize_scale_to_200m_boundary[1] -
        #             self.sat_dataset.satimgsize_scale_to_200m_boundary[0])
        normalized_scales = (self.sat_dataset.satimgsize_scale_to_200m_boundary[1] - scales) / (
                self.sat_dataset.satimgsize_scale_to_200m_boundary[1] -
                self.sat_dataset.satimgsize_scale_to_200m_boundary[0])
        center_indices = (normalized_scales * (n_gird_lod - 1)).squeeze()
        m_indices = torch.arange(n_gird_lod, device=normalized_scales.device)  # 形状: [M]

        dist = torch.abs(center_indices.unsqueeze(1) - m_indices.unsqueeze(0))
        epsilon = 1e-8
        # scale_weights = 1.0 / (dist + epsilon)
        p = 0.5  # p=2时，权重随距离二次方衰减（衰减更快）;p=0.5时，衰减更慢
        scale_weights = 1.0 / (dist.pow(p) + epsilon)
        scale_weights = scale_weights / torch.sum(scale_weights, dim=-1, keepdim=True)

        feats_grid = (feats_grid.reshape(gird_3d_coords.shape[0], n_gird_lod, -1) * scale_weights.unsqueeze(
            -1).to(self.device)).sum(dim=-2)

        if len(coords.shape) == 3:
            feats_grid = feats_grid.reshape(n, m, feats_grid.shape[-1])
        return feats_grid

    def _get_feats_fm_imgs(self, satimgs_flatten, feat_fm_agg=True):
        batchsize = 512
        feat_galllery = []
        for batch in torch.split(satimgs_flatten, batchsize, dim=0):
            feat = self.img_encoder(batch.to(self.device))
            if feat_fm_agg:
                if self.agg_name == 'salad':
                    feat = self.img_aggregator.forward(feat)
                elif self.agg_name == 'g2m':
                    feat = self.img_aggregator.forward(
                        feat[:, 1:, :].permute(0, 2, 1).reshape(feat.shape[0], self.feat_q_dim, 14, 14), normalize=True)
            else:
                feat = feat[:, 0, :]
            feat_galllery.append(feat.detach().cpu())
        feat_galllery = torch.cat(feat_galllery)
        return feat_galllery

    def _get_feat_gallery_fm_grid(self, overlap=0.5, delta_rot_rangle=10, scale=None):
        with torch.no_grad():
            # construct coords_gallery
            scale = self.sat_dataset.satimgsize_scale_to_200m_mean if scale is None else scale
            satimgsize2crop = scale*200./self.sat_dataset.geo_res_m
            nrcs_gallery = self.sat_dataset.crop_sat_unifrom(size2clip=satimgsize2crop, overlap=overlap, only_nrcs=True)
            nrcs_gallery_flatten = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)
            rots_angle = [-180 + delta_rot_rangle * i for i in range(360 // delta_rot_rangle)]
            rots_rad = torch.tensor(np.deg2rad(np.stack(rots_angle)), dtype=torch.float32)
            coords_gallery = torch.concatenate([nrcs_gallery_flatten.unsqueeze(1).expand(-1, rots_rad.shape[0], -1),
                                                rots_rad[None, :, None].expand(nrcs_gallery_flatten.shape[0], -1, 1),
                                                torch.ones([nrcs_gallery_flatten.shape[0], rots_rad.shape[0], 1],
                                                           dtype=torch.float32) * scale
                                                ], dim=-1)
            coords_gallery_flatten = coords_gallery.flatten(start_dim=0, end_dim=1)
            coords_gallery_encoded_flatten = encode_4d_coords(coords_gallery_flatten,
                                                              rc_encoder=self.rc_pos_encoder,
                                                              rot_endcoder=self.rot_pos_encoder,
                                                              scale_encoder=self.scale_pos_encoder)
            gallery_shape = torch.Size([nrcs_gallery.shape[0],nrcs_gallery.shape[1],rots_rad.shape[0]])

            # construct feat_gallery form grid
            feat_gallery_flatten = []
            chunk_size = 512  # 定义块大小
            coords_chunks = torch.split(coords_gallery_flatten, chunk_size)
            encoded_coords_chunks = torch.split(coords_gallery_encoded_flatten, chunk_size)
            for coords_4d, encoded_coords_4d in zip(coords_chunks, encoded_coords_chunks):
                feat = self._feats_fm_grid(coords_4d)
                feat = self.grid_mlp(inputs=feat, condition_features=encoded_coords_4d.to(feat.device))
                feat_gallery_flatten.append(feat.detach().cpu())
            feat_gallery_flatten = torch.concatenate(feat_gallery_flatten, dim=0)
            feat_gallery_flatten = torch.nn.functional.normalize(feat_gallery_flatten, dim=-1, p=2)
            feat_gallery = feat_gallery_flatten.reshape(*nrcs_gallery.shape[:2], rots_rad.shape[0], -1)
            print(f"Gallery shape: {feat_gallery.shape}")  # [68, 50, 36, 1024]
            print(f"Total candidates: {feat_gallery_flatten.shape[0]}")  # 122,400

            dict2ret = {
                'gallery_shape': gallery_shape,
                'feat_gallery_flatten': feat_gallery_flatten,
                'nrc_gallery': nrcs_gallery,
                "rots_rad": rots_rad,
                "scale": scale,
                "coords_gallery_flatten": coords_gallery_flatten,
                'coords_gallery_encoded_flatten':coords_gallery_encoded_flatten
            }
            return dict2ret

    def test_xy(self):
        self._test_ready()
        from util_vis_4d_fields_in_2d import FieldViser2D
        self.viser_2d = FieldViser2D()

        with torch.no_grad():
            # get satimgs_gallery:
            # get satimgs_gallery,v0:
            # coords_gallery = self.viser_2d.mk_grid_nrcs(
            #     row_range=self.sat_dataset.nr2sample_range,
            #     col_range=self.sat_dataset.nc2sample_range,
            #     resolution=64,
            #     d_val=0,
            #     s_val=self.sat_dataset.satimgsize_scale_to_200m_mean,
            # )
            # satimgs_gallery = self.sat_dataset.crop_satimgs_by_4d_coords(coords_gallery)
            # satimgs_gallery_flatten = satimgs_gallery.flatten(start_dim=0,end_dim=1)
            # get satimgs_gallery,v1:
            overlap = 0.5
            sat_tiles, nrcs_grid = self.sat_dataset.crop_sat_unifrom(size2clip=self.sat_dataset.satimgsize2crop_mean,
                                                                     overlap=overlap)
            nrcs_grid_flatten = torch.from_numpy(nrcs_grid).flatten(start_dim=0, end_dim=1)
            coords_gallery_flatten = torch.cat([nrcs_grid_flatten, torch.zeros(nrcs_grid_flatten.shape[0], 1),
                                                torch.ones(nrcs_grid_flatten.shape[0],
                                                           1) * self.sat_dataset.satimgsize_scale_to_200m_mean], dim=1)
            satimgs_gallery_flatten = self.sat_dataset.scale_transform(sat_tiles.flatten(start_dim=0, end_dim=1))

            nrcs_q = torch.from_numpy(self.sat_dataset.mk_rand_nrcs(128))
            coords_q = torch.concatenate([nrcs_q, torch.zeros(nrcs_q.shape[0], 1), torch.ones(nrcs_q.shape[0],
                                                                                              1) * self.sat_dataset.satimgsize_scale_to_200m_mean],
                                         dim=-1)
            satimgs_q = self.sat_dataset.crop_satimgs_by_4d_coords(coords_q)

            # feat_fm_agg = True
            # batchsize=256
            # feat_galllery = []
            # for batch in torch.split(satimgs_gallery_flatten, batchsize,dim=0):
            #     feat = self.img_encoder(batch.to(self.device))
            #     if feat_fm_agg:
            #         feat = self.img_aggregator(feat[:, 1:, :].permute(0, 2, 1).reshape(feat.shape[0], self.feat_q_dim, 14, 14))
            #     else:
            #         feat = feat[:,0,:]
            #     feat_galllery.append(feat.detach().cpu())
            # feat_galllery = torch.cat(feat_galllery)
            # feat_q = self.img_encoder(satimgs_q.to(self.device))
            # if feat_fm_agg:
            #     feat_q = self.img_aggregator(feat_q[:, 1:, :].permute(0, 2, 1).reshape(feat_q.shape[0], self.feat_q_dim, 14, 14)).detach().cpu()
            # else:
            #     feat_q = feat_q[:, 0, :].detach().cpu()
            def get_feats(feat_fm_agg=True):
                batchsize = 256
                feat_galllery = []
                for batch in torch.split(satimgs_gallery_flatten, batchsize, dim=0):
                    feat = self.img_encoder(batch.to(self.device))
                    if feat_fm_agg:
                        feat = self.img_aggregator(
                            feat[:, 1:, :].permute(0, 2, 1).reshape(feat.shape[0], self.feat_q_dim, 14, 14))
                    else:
                        feat = feat[:, 0, :]
                    feat_galllery.append(feat.detach().cpu())
                feat_galllery = torch.cat(feat_galllery)
                feat_q = self.img_encoder(satimgs_q.to(self.device))
                if feat_fm_agg:
                    feat_q = self.img_aggregator(
                        feat_q[:, 1:, :].permute(0, 2, 1).reshape(feat_q.shape[0], self.feat_q_dim, 14,
                                                                  14)).detach().cpu()
                else:
                    feat_q = feat_q[:, 0, :].detach().cpu()
                return feat_galllery, feat_q

            feat_galllery_agg, feat_q_agg = get_feats(feat_fm_agg=True)
            # feat_galllery_dino, feat_q_dino = get_feats(feat_fm_agg=False)

            pred_agg = torch.norm(feat_q_agg.unsqueeze(1) - feat_galllery_agg.unsqueeze(0), dim=-1, p=2)
            pred_v_agg, pred_id_agg = torch.sort(pred_agg.squeeze(), dim=-1, descending=False)
            # pred_dino = torch.norm(feat_q_dino.unsqueeze(1) - feat_galllery_dino.unsqueeze(0), dim=-1, p=2)
            # pred_v_dino, pred_id_dino = torch.sort(pred_dino.squeeze(), dim=-1, descending=False)

            from eval_recall_fm_salad import qurey_label_fm_gallery_nrc, compute_recall_by_label
            distrc_sats2uav, uav_labels_query = qurey_label_fm_gallery_nrc(nrcs_grid_flatten, nrcs_q,
                                                                           self.sat_dataset.halfimg_radius_nrc)
            compute_recall_by_label(uav_labels_query, pred_id_agg[:, :200], k_values=[1, 5, 10, 20, 50])

            vis = False
            if vis:
                gt2gallery_dist_mat = torch.norm(nrcs_q.unsqueeze(1) - nrcs_grid_flatten.unsqueeze(0), dim=-1, p=2)
                gt_ids_flatten = torch.argmin(gt2gallery_dist_mat, dim=-1)
                gt_rows = gt_ids_flatten // nrcs_grid.shape[1]
                gt_cols = gt_ids_flatten % nrcs_grid.shape[1]
                gt_ids = torch.stack([gt_rows, gt_cols], dim=0).T
                gt_ids_np = gt_ids.detach().cpu().numpy()

                id_seled = 0
                from util_vis_retrieval_in_2d import visualize_response_map, visualize_response_map_3d
                visualize_response_map(
                    response_map=pred_agg[id_seled].reshape(sat_tiles.shape[:2]).detach().cpu().numpy(),
                    ground_truth_idx=(gt_ids_np[id_seled][0], gt_ids_np[id_seled][1]),
                    mark_extreme='min',
                    # cmap = 'coolwarm',
                )

    def test_xy_rot_fm_metric_net(self):
        self._test_ready()
        with torch.no_grad():
            # construct coords_gallery
            overlap = 0.5
            nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
                size2clip=self.sat_dataset.satimgsize2crop_mean, overlap=overlap, only_nrcs=True)
            nrcs_gallery_flatten = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)
            delta_rot_rangle = 10
            rot_angles = [-180 + delta_rot_rangle * i for i in range(360 // delta_rot_rangle)]
            rots_ref = torch.tensor(np.deg2rad(np.stack(rot_angles)), dtype=torch.float32)
            coords_gallery = torch.concatenate([nrcs_gallery_flatten.unsqueeze(1).expand(-1, rots_ref.shape[0], -1),
                                                rots_ref[None, :, None].expand(nrcs_gallery_flatten.shape[0], -1, 1),
                                                torch.ones([nrcs_gallery_flatten.shape[0], rots_ref.shape[0], 1],
                                                           dtype=torch.float32) * self.sat_dataset.satimgsize_scale_to_200m_mean
                                                ], dim=-1)
            coords_gallery_flatten = coords_gallery.flatten(start_dim=0, end_dim=1)
            coords_gallery_encoded_flatten = encode_4d_coords(coords_gallery_flatten,
                                                              rc_encoder=self.rc_pos_encoder,
                                                              rot_endcoder=self.rot_pos_encoder,
                                                              scale_encoder=self.scale_pos_encoder)

            # construct feat_gallery form grid
            feat_gallery_flatten = []
            chunk_size = 512  # 定义块大小
            coords_chunks = torch.split(coords_gallery_flatten, chunk_size)
            encoded_coords_chunks = torch.split(coords_gallery_encoded_flatten, chunk_size)
            for coords_4d, encoded_coords_4d in zip(coords_chunks, encoded_coords_chunks):
                feat = self._feats_fm_grid(coords_4d)
                feat = self.grid_mlp(inputs=feat, condition_features=encoded_coords_4d.to(feat.device))
                feat_gallery_flatten.append(feat.detach().cpu())
            feat_gallery_flatten = torch.concatenate(feat_gallery_flatten, dim=0)
            feat_gallery_flatten = torch.nn.functional.normalize(feat_gallery_flatten, dim=-1, p=2)
            feat_gallery = feat_gallery_flatten.reshape(*nrcs_gallery.shape[:2], rots_ref.shape[0], -1)
            print(f"Gallery shape: {feat_gallery.shape}")  # [68, 50, 36, 1024]
            print(f"Total candidates: {feat_gallery_flatten.shape[0]}")  # 122,400

            # get query_feat from satimgs
            n_query = 32
            nrcs_q = torch.from_numpy(self.sat_dataset.mk_rand_nrcs(n_query))
            rots_q = -torch.pi + 2 * torch.pi * torch.rand(n_query)
            coords_q = torch.concatenate([nrcs_q, rots_q.unsqueeze(1), torch.ones(nrcs_q.shape[0],
                                                                                  1) * self.sat_dataset.satimgsize_scale_to_200m_mean],
                                         dim=-1)
            satimgs_q = self.sat_dataset.crop_satimgs_by_4d_coords(coords_q)
            feat_q = self._get_feats_fm_imgs(satimgs_q, feat_fm_agg=True)

            # ===== 双重分块检索  =====
            n_query = feat_q.shape[0]
            n_gallery = feat_gallery_flatten.shape[0]
            query_chunk_size = 4
            gallery_chunk_size = 2048
            device = next(self.metric_net.parameters()).device
            # 存储所有距离
            all_distances = []
            # 对query分块
            for q_start in range(0, n_query, query_chunk_size):
                q_end = min(q_start + query_chunk_size, n_query)
                query_chunk = feat_q[q_start:q_end].to(device)  # [Q, D]
                Q = query_chunk.shape[0]
                # 对该query chunk，收集所有gallery的距离
                query_distances = []
                for g_start in range(0, n_gallery, gallery_chunk_size):
                    g_end = min(g_start + gallery_chunk_size, n_gallery)
                    gallery_chunk = feat_gallery_flatten[g_start:g_end].to(device)
                    coords_chunk = coords_gallery_encoded_flatten[g_start:g_end].to(device)
                    G = gallery_chunk.shape[0]
                    # 构造配对
                    query_expanded = query_chunk.unsqueeze(1).expand(Q, G, -1)
                    gallery_expanded = gallery_chunk.unsqueeze(0).expand(Q, G, -1)
                    coords_expanded = coords_chunk.unsqueeze(0).expand(Q, G, -1)
                    # 计算距离
                    distances = self.metric_net(
                        query_expanded,
                        gallery_expanded,
                        coords_expanded
                    )  # [Q, G]
                    query_distances.append(distances.cpu())
                    print(f"Processed query [{q_start}:{q_end}], gallery [{g_start}:{g_end}]")
                # 拼接该query chunk的所有距离
                query_distances = torch.cat(query_distances, dim=1)  # [Q, n_gallery]
                all_distances.append(query_distances)
            # 拼接所有query的距离
            all_distances = torch.cat(all_distances, dim=0)  # [n_query, n_gallery]
            print(f"\nDistance matrix shape: {all_distances.shape}")

            # 排序预测结果
            pred_dists, pred_indices = torch.sort(all_distances, dim=-1, descending=False)
            # 取Top-K
            topk = 50
            best_distances = pred_dists[:, :topk]  # [n_query, topk]
            best_indices = pred_indices[:, :topk]  # [n_query, topk]
            # 将预测的索引转换回4D坐标
            pred_coords_flatten = coords_gallery_flatten[best_indices]  # [n_query, 4]
            gallery_shape = torch.Size([*nrcs_gallery.shape[:2], rots_ref.shape[0]])
            from util_unravel_index import unravel_index
            pred_ids_unraled = unravel_index(best_indices, gallery_shape)
            pred_id_nrc_flatten = best_indices // rots_ref.shape[0]

            # computing gt_index
            gt2gallery_rc_dist_mat = torch.norm(nrcs_q.unsqueeze(1) - nrcs_gallery_flatten.unsqueeze(0), dim=-1, p=2)
            gt_ids_rc_flatten = torch.argmin(gt2gallery_rc_dist_mat, dim=-1)
            gt_rows = gt_ids_rc_flatten // nrcs_gallery.shape[1]
            gt_cols = gt_ids_rc_flatten % nrcs_gallery.shape[1]
            gt_ids_rc = torch.stack([gt_rows, gt_cols], dim=0).T
            gt_ids_rc_np = gt_ids_rc.detach().cpu().numpy()
            # angular_diff_raw = torch.tensor(np.deg2rad(np.stack(rot_angles)),dtype=torch.float32,device=rots_q.device)[None,...]-rots_q.unsqueeze(1)
            if rots_ref.max() < 4:
                angular_diff_raw = rots_ref[None, ...] - rots_q.unsqueeze(1)
            else:
                angular_diff_raw = torch.deg2rad(rots_ref)[None, ...] - rots_q.unsqueeze(1)
            angular_diff_normalized = (angular_diff_raw + torch.pi) % (
                        2 * torch.pi) - torch.pi  # 将差值归一化到 [-pi, pi] 这个区间内,找到最短角度差,公式为: (diff + pi) % (2 * pi) - pi
            angular_dist = torch.abs(angular_diff_normalized)
            gt_ids_rot = torch.argmin(angular_dist, dim=-1)  # Shape: (N_q)
            gt_ids_rot_np = gt_ids_rot.detach().cpu().numpy()

            # compute the recall
            from eval_recall_fm_salad import qurey_label_fm_gallery_nrc, compute_recall_by_label, create_success_mask
            distrc_sats2uav, uav_labels_query = qurey_label_fm_gallery_nrc(nrcs_gallery_flatten, nrcs_q,
                                                                           self.sat_dataset.halfimg_radius_nrc)
            compute_recall_by_label(uav_labels_query, pred_id_nrc_flatten, k_values=[1, 5, 10, 20, 50])

            rot2eavl_mask = create_success_mask(pred_id_nrc_flatten[:, :1], uav_labels_query, k=1)
            rot_recall_results = compute_recall_by_label(
                q_labels=gt_ids_rot[rot2eavl_mask].unsqueeze(-1),
                pred_labels_per_query=pred_ids_unraled[2][rot2eavl_mask, :10],
                k_values=[1, 3, 5, 10],
                title="Conditional Rotation Recall (based on radius search)"
            )
            print("\n条件旋转召回率结果:", rot_recall_results)

            diff_pred_min = torch.abs(pred_coords_flatten[:, 0, :] - coords_q)[:, :3]
            pred_nrcdist2gt_min = torch.norm(diff_pred_min[:, :2], dim=-1)

        vis = False
        if vis:
            pred_dists_shaped = all_distances.reshape(pred_dists.shape[0], *nrcs_gallery.shape[:2], -1)
            id_seled = 0
            map2vis = pred_dists_shaped[id_seled][:, :, gt_ids_rot[0]]
            # map2vis = pred_dists_shaped[id_seled].reshape(gallery_shape)[:, :, gt_ids_rot[0]]
            from util_vis_retrieval_in_2d import visualize_response_map, visualize_response_map_3d
            visualize_response_map(
                response_map=map2vis.detach().cpu().numpy(),
                ground_truth_idx=(gt_ids_rc_np[id_seled][0], gt_ids_rc_np[id_seled][1]),
                mark_extreme='min',
                # cmap = 'coolwarm',
            )
            visualize_response_map_3d(torch.exp(-map2vis))

    def test_xy_rot(self):
        self._test_ready()
        from util_vis_4d_fields_in_2d import FieldViser2D
        self.viser_2d = FieldViser2D()

        with torch.no_grad():
            overlap = 0.5
            suffix = f"_overlap{overlap}_radius{self.sat_dataset.halfimg_radius_meter:.0f}m.mat"
            p2gallery_mat = f"{os.path.dirname(self.opt.load2test)}/{os.path.basename(self.opt.load2test).replace('.pth', suffix)}"

            if os.path.exists(p2gallery_mat):
                gallery_mat = scipy.io.loadmat(p2gallery_mat)
                feat_gallery_roted = torch.tensor(gallery_mat['feat_gallery'], dtype=torch.float32).flatten(start_dim=0,
                                                                                                            end_dim=1)
                nrcs_gallery = torch.tensor(gallery_mat['nrcs_gallery'], dtype=torch.float32)
                rots_ref = torch.tensor(gallery_mat['rots_ref'], dtype=torch.float32).squeeze()
                nrcs_gallery_flatten = nrcs_gallery.flatten(start_dim=0, end_dim=1)
            else:
                nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
                    size2clip=self.sat_dataset.satimgsize2crop_mean, overlap=overlap, only_nrcs=True)
                nrcs_gallery_flatten = torch.tensor(nrcs_gallery, dtype=torch.float32).flatten(start_dim=0, end_dim=1)
                delta_rot_rangle = 10
                rot_angles = [-180 + delta_rot_rangle * i for i in range(360 // delta_rot_rangle)]
                rots_ref = torch.tensor(np.deg2rad(np.stack(rot_angles)), dtype=torch.float32)
                coords_gallery = torch.concatenate([nrcs_gallery_flatten.unsqueeze(1).expand(-1, rots_ref.shape[0], -1),
                                                    rots_ref[None, :, None].expand(nrcs_gallery_flatten.shape[0], -1,
                                                                                   1),
                                                    torch.ones([nrcs_gallery_flatten.shape[0], rots_ref.shape[0], 1],
                                                               dtype=torch.float32) * self.sat_dataset.satimgsize_scale_to_200m_mean
                                                    ], dim=-1)
                coords_gallery_flatten = coords_gallery.flatten(start_dim=0, end_dim=1)
                coords_gallery_encoded_flatten = encode_4d_coords(coords_gallery_flatten,
                                                                  rc_encoder=self.rc_pos_encoder,
                                                                  rot_endcoder=self.rot_pos_encoder,
                                                                  scale_encoder=self.scale_pos_encoder)
                feat_gallery_flatten = []
                chunk_size = 512  # 定义块大小
                # 将两个张量都进行切分
                coords_chunks = torch.split(coords_gallery_flatten, chunk_size)
                encoded_coords_chunks = torch.split(coords_gallery_encoded_flatten, chunk_size)
                for coords_4d, encoded_coords_4d in zip(coords_chunks, encoded_coords_chunks):
                    feat = self._feats_fm_grid(coords_4d)
                    feat = self.grid_mlp(inputs=feat, condition_features=encoded_coords_4d.to(feat.device))
                    feat_gallery_flatten.append(feat.detach().cpu())
                feat_gallery_flatten = torch.concatenate(feat_gallery_flatten, dim=0)
                feat_gallery_flatten = torch.nn.functional.normalize(feat_gallery_flatten, dim=-1, p=2)
                feat_gallery = feat_gallery_flatten.reshape(*nrcs_gallery.shape[:2], rots_ref.shape[0], -1)

                result = {'feat_gallery': feat_gallery.detach().cpu().numpy(),
                          'nrcs_gallery': nrcs_gallery,
                          'rots_ref': rots_ref.detach().cpu().numpy(),
                          }
                scipy.io.savemat(p2gallery_mat, result)

            # get satimgs_query_feat
            n_query = 512
            nrcs_q = torch.from_numpy(self.sat_dataset.mk_rand_nrcs(n_query))
            rots_q = -torch.pi + 2 * torch.pi * torch.rand(n_query)
            coords_q = torch.concatenate([nrcs_q, rots_q.unsqueeze(1), torch.ones(nrcs_q.shape[0],
                                                                                  1) * self.sat_dataset.satimgsize_scale_to_200m_mean],
                                         dim=-1)
            satimgs_q = self.sat_dataset.crop_satimgs_by_4d_coords(coords_q)
            feat_q = self._get_feats_fm_imgs(satimgs_q, feat_fm_agg=True)

            # computing pred
            # pred_agg = torch.norm(feat_q[:,None,None,:] - feat_gallery_roted.unsqueeze(0), dim=-1, p=2)
            # pred_v,pred_id = torch.sort( pred_agg.flatten(start_dim=1,end_dim=2), dim=-1, descending=False)
            pred_agg = torch.norm(feat_q.unsqueeze(1) - feat_gallery_flatten.unsqueeze(0), dim=-1, p=2)
            pred_v, pred_id = torch.sort(pred_agg, dim=-1, descending=False)
            gallery_shape = torch.Size([*nrcs_gallery.shape[:2], rots_ref.shape[0]])
            pred_id_rot = pred_id % gallery_shape[2]  # 应该是 pred_id % R
            pred_id_rc = pred_id // gallery_shape[2]  # 应该是 pred_id // R
            pred_ids_unraled = unravel_index(pred_id, gallery_shape)

            from eval_recall_fm_salad import qurey_label_fm_gallery_nrc, compute_recall_by_label, create_success_mask
            distrc_sats2uav, uav_labels_query = qurey_label_fm_gallery_nrc(nrcs_gallery_flatten, nrcs_q,
                                                                           self.sat_dataset.halfimg_radius_nrc)
            compute_recall_by_label(uav_labels_query, pred_id_rc[:, :200], k_values=[1, 5, 10, 20, 50])

            # computing gt
            gt2gallery_rc_dist_mat = torch.norm(nrcs_q.unsqueeze(1) - nrcs_gallery_flatten.unsqueeze(0), dim=-1, p=2)
            gt_ids_rc_flatten = torch.argmin(gt2gallery_rc_dist_mat, dim=-1)
            gt_rows = gt_ids_rc_flatten // nrcs_gallery.shape[1]
            gt_cols = gt_ids_rc_flatten % nrcs_gallery.shape[1]
            gt_ids_rc = torch.stack([gt_rows, gt_cols], dim=0).T
            gt_ids_rc_np = gt_ids_rc.detach().cpu().numpy()
            # angular_diff_raw = torch.tensor(np.deg2rad(np.stack(rot_angles)),dtype=torch.float32,device=rots_q.device)[None,...]-rots_q.unsqueeze(1)
            if rots_ref.max() < 4:
                angular_diff_raw = rots_ref[None, ...] - rots_q.unsqueeze(1)
            else:
                angular_diff_raw = torch.deg2rad(rots_ref)[None, ...] - rots_q.unsqueeze(1)
            angular_diff_normalized = (angular_diff_raw + torch.pi) % (
                        2 * torch.pi) - torch.pi  # 将差值归一化到 [-pi, pi] 这个区间内,找到最短角度差,公式为: (diff + pi) % (2 * pi) - pi
            angular_dist = torch.abs(angular_diff_normalized)
            gt_ids_rot = torch.argmin(angular_dist, dim=-1)  # Shape: (N_q)
            gt_ids_rot_np = gt_ids_rot.detach().cpu().numpy()

            rot_mask = create_success_mask(pred_id_rc[:, :1], uav_labels_query, k=1)
            rot_recall_results = compute_recall_by_label(
                q_labels=gt_ids_rot[rot_mask].unsqueeze(-1),
                pred_labels_per_query=pred_ids_unraled[2][rot_mask, :10],
                k_values=[1, 3, 5, 10],
                title="Conditional Rotation Recall (based on radius search)"
            )
            print("\n条件旋转召回率结果:", rot_recall_results)

            gtrot2eval = rots_q[rot_mask]
            predrot2eval = rots_ref[pred_ids_unraled[2][rot_mask, 0]]
            rot_err2eval = gtrot2eval - predrot2eval
            rot_err_min = torch.abs(torch.atan2(torch.sin(rot_err2eval), torch.cos(rot_err2eval)))
            rot_err_min = torch.rad2deg(rot_err_min)
            print("\n平均旋转估计误差:", rot_err_min.mean().item())

            compute_udf = False
            if compute_udf:
                pred_rc_err_min = torch.norm(
                    nrcs_q - nrcs_gallery[pred_ids_unraled[0][:, 0], pred_ids_unraled[1][:, 0]], dim=-1, p=2)
                pred_rot_err_min = rots_q - rots_ref[pred_id_rot[:, 0]]
                pred_rot_err_min = torch.abs(torch.atan2(torch.sin(pred_rot_err_min), torch.cos(
                    pred_rot_err_min))).squeeze()  # atan2 函数的输出范围是 [-π, π]
                self.udf_compter = UDFComputer(self.sat_dataset)
                udf_dist = self.udf_compter.compute_udf_fm_diff(pred_rc_err_min, pred_rot_err_min)

            vis = False
            if vis:
                gt2gallery_dist_mat = torch.norm(nrcs_q.unsqueeze(1) - nrcs_gallery_flatten.unsqueeze(0), dim=-1, p=2)
                gt_ids_flatten = torch.argmin(gt2gallery_dist_mat, dim=-1)
                gt_rows = gt_ids_flatten // nrcs_gallery.shape[1]
                gt_cols = gt_ids_flatten % nrcs_gallery.shape[1]
                gt_ids = torch.stack([gt_rows, gt_cols], dim=0).T
                gt_ids_np = gt_ids.detach().cpu().numpy()

                id_seled = 0
                from util_vis_retrieval_in_2d import visualize_response_map, visualize_response_map_3d
                map2vis = pred_agg[id_seled].reshape(gallery_shape)[:, :, gt_ids_rot[0]]
                visualize_response_map(
                    response_map=map2vis.detach().cpu().numpy(),
                    ground_truth_idx=(gt_ids_np[id_seled][0], gt_ids_np[id_seled][1]),
                    mark_extreme='min',
                    # cmap = 'coolwarm',
                )
                visualize_response_map_3d(torch.exp(-map2vis))
                # visualize_response_map_3d( -map2vis )

    def mk_gallery_feat(self):
        self._test_ready()
        satimgsize2crop = self.sat_dataset.satimgsize2crop
        overlap = 0.5
        sat_tiles, nrcs_girdcoord_center = self.sat_dataset.crop_sat_unifrom(size2clip=satimgsize2crop, overlap=overlap)

        from train_img_encoder.datasets_custom.util_mk_data_transform import RandomRotationWithAngle
        rotater = RandomRotationWithAngle(degrees=180, same_on_batch=True)
        sat_tiles_roted = []
        delta_rot_rangle = 10
        rot_angles = [-180 + delta_rot_rangle * i for i in range(360 // delta_rot_rangle)]
        for rot in rot_angles:
            sat_tiles_roted.append(rotater(sat_tiles.flatten(start_dim=0, end_dim=1), rot))
        sat_tiles_roted = torch.concatenate(sat_tiles_roted, dim=0)
        sat_tiles_roted = self.sat_dataset.scale_transform(sat_tiles_roted)

        batchsize = 512
        feat_gallery = []
        with torch.no_grad():
            for batch in torch.split(sat_tiles_roted, batchsize, dim=0):
                feats = self.img_encoder(batch[:, 0, ...].to(self.device))
                feats = self.img_aggregator(feats[:, 1:, :].permute(0, 2, 1).reshape(-1, self.feat_q_dim, 14, 14))
                feat_gallery.append(feats)
        feat_gallery = torch.cat(feat_gallery, dim=0)
        feat_gallery = feat_gallery.reshape(-1, 360 // delta_rot_rangle, feat_gallery.shape[-1])

        # save the result matrix:
        result = {'feat_gallery': feat_gallery.detach().cpu().numpy(),
                  'nrc_gallery': nrcs_girdcoord_center.detach().cpu().numpy(),
                  }
        suffix = f"_overlap{overlap}_radius{self.sat_dataset.halfimg_radius_meter:.0f}m.mat"
        scipy.io.savemat(
            f"{os.path.dirname(self.opt.load2test)}/{os.path.basename(self.opt.load2test).replace('.pth', suffix)}",
            result)


if __name__ == '__main__':
    torch.manual_seed(666)
    np.random.seed(2025)
    trainer = Trainer()
    trainer.train()
    # trainer.test_xy_rot_fm_metric_net()
    # trainer.test_xy_rot()
    # trainer.mk_map_feats()
    # trainer.test_xy()
    # trainer.output_test_res()
    # trainer.test_rot()
    # trainer.test_radon_wo_translate()
    # trainer.test_radon_wo_translate_crossdomain()