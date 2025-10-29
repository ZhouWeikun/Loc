# -*- coding: utf-8 -*-
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from __future__ import print_function, division
import argparse
import torch
import tqdm
# from torch.ao.nn.quantized.functional import threshold
import numpy as np
import time
# from triton.language import dtype
# from tool.utils import load_network_wstate, save_network_wstate
import warnings
from torch.utils.tensorboard import SummaryWriter
import glob

warnings.filterwarnings("ignore")
import yaml
import os
import json
#custom:
# from datasets.make_dataloader import make_dataloader
# from datasets_custom.make_dataloder_classify import make_dataloader_train
# from datasets_custom.make_dataloader_xmu import make_dataloader_xmu
# from datasets_custom.make_dataloader_gta import make_dataloader_gta
# from datasets_custom.make_dataloader_wingtra import make_dataloader_wingtra
# from exps.exp24.datasets_custom.make_dataloader_dsalad import  make_dataloader
from train_img_encoder.nets_taskflow import make_img_encoder
from tool.utils import get_logger, get_unique_exp_dir
from dataset_wingtra_4d import UAVDataset,SatDataset
from util_udf_computer import UDFComputer


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    #about exp setting
    parser.add_argument('--exps_dir', default='.exps/',type=str, help='the dir that save experiments')
    parser.add_argument('--exp_name', default='debug',type=str, help='the experiment name that will be saved in exps dir in the root')
    parser.add_argument('--p_yaml', default='/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/opts_wingtra.yaml', type=str, help='the yaml file about the defult setting')
    parser.add_argument('--p_satinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu__512h_lineClassed/dataset_xmu_meta/satimg_xiangan_xmu_info.json',
                        type=str, help='')
    parser.add_argument('--p_uavinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu_512h_lineClassed/dataset_xmu_meta/uavimgs_xiangan_xmu_info.json',
                        type=str, help='')
    parser.add_argument('--p_uav_geocsv',
                        default='/home/data/zwk/data_uavimgs_wingtra/Zurich/IMAGES_info/uavimgs_geo_corrected_v1.csv',
                        type=str, help='')
    parser.add_argument('--dataset_name',default='xmu', type=str)
    parser.add_argument('--load2test', default="/home/data/zwk/pyproj_neuloc_v0/exps/exp_wohead_vit-b/epoch000.pth", type=str, help='path for testing') # for testing
    parser.add_argument('--load2train', default="", type=str, help='exps path for pre-loading') #for continuing training
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--val', action='store_true', default = False)
    parser.add_argument('--val_freq', default = 10, type=int )
    parser.add_argument('--tensorboard', action='store_true', default = True)
    parser.add_argument('--n_satrand_per_uav', default=8, type=int, help='will be used in dataset')
    #about hardware
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--num_worker', default = 16, type=int, help='')
    parser.add_argument('--batchsize_sat', default = 32, type=int, help='batchsize')
    parser.add_argument('--batchsize_uav', default = 32, type=int, help='batchsize')
    parser.add_argument('--autocast', action='store_true', default=True, help='use mix precision')
    #about data setting,version 2:
    parser.add_argument('--imgsize2net', default=224, type=int)
    parser.add_argument('--satimgsize2crop', default=224, type=int)
    # parser.add_argument('--n_rand2sample_per_pos', default=2, type=int)
    # parser.add_argument('--uav_da', nargs='+', default=['rr'],help='rr=random_rotate,ra=random affine,re=random erasing,cj=color jitter,cda=color data argument')
    # parser.add_argument('--sat_da', nargs='+', default=['ra','re'],help='rr=random_rotate,ra=random affine,re=random erasing,cj=color jitter,cda=color data argument')
    # parser.add_argument('--erasing_p', default=0.3, type=float,help='random erasing probability, in [0,1]')
    #about networks
    parser.add_argument('--backbone', default="ViTB-384", type=str, help='ViTB-224;ViTS-224;dinov2_vitb14;ViTB-384')
    parser.add_argument('--head', default="", type=str, help='salad;FSRA;LPN;') #"" means no head
    parser.add_argument('--block', default=2, type=int, help='') #will by used when headF=FSRA,LPN,NetVLAD,NeXtVLAD
    parser.add_argument('--num_bottleneck', default=512, type=int, help='the dimensions for embedding the feature')
    parser.add_argument('--head_pool', default="avg", type=str, help='head pooling type for applying') #will by used when head=SingleBranch
    parser.add_argument('--wcls_token', default=False, type=bool) #will by used when head=SingleBranch
    parser.add_argument('--norm_output', default=True, type=bool)
    parser.add_argument('--w_classify', default=False, action='store_true', help='')
    parser.add_argument('--feature_loss', nargs='+', default=["WeightedSoftTripletLoss"],
                        help='"InfoNceLoss","MSLoss","TripletLoss","HardMiningTripletLoss","SameDomainTripletLoss","WeightedSoftTripletLoss","ContrastiveLoss"')
    parser.add_argument('--cls_loss', default="CELoss", type=str, help='loss type of representation learning')
    parser.add_argument('--kl_loss', default="KLLoss", type=str, help='loss type of mutual learning')


    #about learning setting
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
    yaml_file_path = opt.p_yaml # 获取 YAML 文件路径
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
        'exp_setting': ['p_yaml', 'p_satinfo_json', 'p_uavinfo_json','p_uav_geocsv','exp_name','exps_dir','load2train', 'load2test', 'val','val_freq', 'save_freq', 'tensorboard'],
        'hardware_setting': ['gpu_ids', 'num_worker', 'batchsize_sat','batchsize_uav', 'autocast'],
        'data_setting': ['imgsize2net', 'satimgsize2crop', ],
        'learning_setting': ['warm_epoch', 'num_epochs', 'droprate','optimizer','lr_sched'],
        'network_setting': ['w_classify','block', 'cls_loss', 'feature_loss', 'kl_loss', 'num_bottleneck', 'backbone', 'head', 'head_pool','wcls_token','norm_output'] # 补上 wcls_token 和 norm_output
    }
    opt.group_dict = group_info
    print(opt) # 打印最终的参数

    return opt


class Trainer(object):
    def __init__(self):
        self.opt = get_parse()

        if torch.cuda.is_available():
            device = torch.device("cuda:"+self.opt.gpu_ids[0])
            self.opt.use_gpu = True
        else:
            device = torch.device("cpu")
            self.opt.use_gpu = False
        self.device = device

        # config the img_encoder
        self.vis_encoder = make_img_encoder(self.opt).to(self.device)
        feat_q_dim = self.vis_encoder.backbone.output_channel
        self.feat_q_dim = feat_q_dim
            # freeze the para
        for param in self.vis_encoder.parameters():
            param.requires_grad = False
            
        # config the aggregator
        # self.decoder = LocalDecoderFiLM(dim=feat_q_dim,c_dim=feat_q_dim,hidden_size=1024,n_blocks=3,output_dim=1,norm_type='none',leaky=True).to(self.device)
        # from models.Head.G2M import G2M
        # self.vis_aggregator = G2M(in_channels=feat_q_dim,out_channels=feat_q_dim,rank=1024).to(self.device)
        # from models.Head.salad import SALAD
        # self.vis_aggregator = SALAD(input_feat_dim=feat_q_dim, global_token_dim=128, pathchsize=16, num_clusters=14, cluster_dim=64).to(self.device)
        from models.Head.salad_residual import SALAD_Residual
        self.vis_aggregator = SALAD_Residual(input_feat_dim=feat_q_dim, base_dim=feat_q_dim, patchsize=16, num_clusters=16, cluster_dim=64).to(self.device)
        # from models.Head.salad_film import SALAD_FiLM
        # self.vis_aggregator = SALAD_FiLM(input_feat_dim=feat_q_dim,base_dim=feat_q_dim,patchsize=16, num_clusters=16, cluster_dim=64).to(self.device)
        self.agg_name = 'salad'
            
        #define the param to save/laod
        self.param2optimize = {
            'vis_aggregator':self.vis_aggregator,
        }
        self.param2freeze = {
            'vis_encoder':self.vis_encoder,
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

        #laod the ckpt
        from tool.util_ckpt_handler import load_param
        load_param(opt.load2test,self.param)
        for k,v in self.param2optimize.items():
            v.eval()
        for k,v in self.param2freeze.items():
            v.eval()

        # config the datalaoder
        from dataset_wingtra_4d import SatDataset
        self.sat_dataset = SatDataset(
            p_satinfo_json=self.opt.p_satinfo_json,
            p_uav_geocsv=self.opt.p_uav_geocsv,
            imgsize2net=224,
        )
        self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset, batch_size = 1,
                                                          num_workers = self.opt.num_worker,
                                                          pin_memory=True, shuffle=True, drop_last=False,
                                                          persistent_workers=True)


    def train(self):
        opt = self.opt

        #config the optimizer ,todo:
        from tool.util_mk_optimizer import create_optimizer_w_temple
        self.optimizer = create_optimizer_w_temple({"vis_encoder":self.vis_encoder,
                                                    'vis_aggregator': self.vis_aggregator,
                                                    # "grid":self.grid,
                                                    # "gird_mlp": self.grid_mlp,
                                                    },'adam')

        # load the ckpt for continuing train if necessray
        from tool.util_ckpt_handler import load_param
        if opt.load2train is not None and len(opt.load2train)>0:
            params_to_add = {"optimizer_state": self.optimizer}
            self.param.update(params_to_add)
            load_param(opt.load2train, self.param)
        begin_epoch = 0

        # config the datalaoders
        self.sat_dataset = SatDataset(
            p_satinfo_json=self.opt.p_satinfo_json,
            p_uav_geocsv=self.opt.p_uav_geocsv,
            imgsize2net=224,
        )
        # self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset, batch_size = self.opt.batchsize_sat,
        #                                                   num_workers = self.opt.num_worker,
        #                                                   pin_memory=True, shuffle=True, drop_last=False,
        #                                                   persistent_workers=True)
        self.uav_datset_trian = UAVDataset(p_uavinfo_json = self.opt.p_uavinfo_json,
                                 geo_res_m=self.sat_dataset.geo_res_m,
                                 trans_georc2nrc_func=self.sat_dataset.transfrom_georc_to_nrc,
                                 stage='train')
        # self.uav_dataloader_train = torch.utils.data.DataLoader(self.uav_datset_trian , batch_size = 128,
        #                                                   num_workers = self.opt.num_worker,
        #                                                   pin_memory=True, shuffle=True, drop_last=True,
        #                                                   persistent_workers=True)
        self.uav_datset_test = UAVDataset(p_uavinfo_json = self.opt.p_uavinfo_json,
                                 geo_res_m=self.sat_dataset.geo_res_m,
                                 trans_georc2nrc_func=self.sat_dataset.transfrom_georc_to_nrc,
                                 stage='test')
        self.uav_dataloader_test = torch.utils.data.DataLoader(self.uav_datset_test , batch_size = 128,
                                                          num_workers = self.opt.num_worker,
                                                          pin_memory=True, shuffle=False, drop_last=False,
                                                          persistent_workers=True)
        # sample neg from sat:
        from dataset_wingtra_4d_uav_sat_pair import UAVSatPairDataset,collate_uav_sat_pair
        from util_sample_neg_nrcs import BoundedNegativeCoordinateSampler
        satmap_sampler = BoundedNegativeCoordinateSampler(self.device)
        self.pair_dataset = UAVSatPairDataset(
            uav_dataset=self.uav_datset_trian,
            sat_dataset=self.sat_dataset,
            satmap_sampler=satmap_sampler,
            device=self.device,
            n_neg_per_sample=1,
        )
        self.dataloader_train = torch.utils.data.DataLoader(self.pair_dataset,
            batch_size=256,num_workers=4,shuffle=True,
            drop_last=True, pin_memory=True, persistent_workers = True,
            collate_fn=collate_uav_sat_pair
        )

        # config the logger&writer
            # make the dir to save the exp
        exp_name = get_unique_exp_dir(opt.exps_dir,opt.exp_name)
        opt.exp_name = exp_name
        exp_dir2save = os.path.join(opt.exps_dir,opt.exp_name)
        os.makedirs(exp_dir2save,exist_ok=True)
            # config the logger
        logger = get_logger("{}/{}/train.log".format(opt.exps_dir,opt.exp_name),'trainer_logger')
        self.logger = logger
        self.logger.info(f"exp ready!, exp_name={exp_name}")
            # config tensorborad if necessary
        self.writer = SummaryWriter("exps/{}/train_tensorboard.log".format(opt.exp_name)) if opt.tensorboard else None

        # config loss
        self.udf_compter = UDFComputer(sat_dataset=self.sat_dataset)
        from losses.WeightedSoftTripletLoss_fm_mat import SWTLoss_fm_mat,MSLoss_fm_mat
        loss_swt = SWTLoss_fm_mat(decoupling=False,)
        loss_ms = MSLoss_fm_mat()
        loss_mse = torch.nn.MSELoss(reduction='mean')
        temperature = 1.
        targets = torch.arange(opt.batchsize_sat).to(self.device)  # [0, 1, 2, ..., B-1]

        # ready to trian
        num_epochs = opt.num_epochs
        since = time.time()
        step = 0
        for epoch in range(begin_epoch,num_epochs):
            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            for it,batch in tqdm.tqdm(enumerate(self.dataloader_train)):
                # uavimgs, coords_q = data
                uavimgs = batch['uav_imgs'].to(self.device)  # [B, C, H, W]                                                                                                                                              │ │
                satimgs_pos = batch['sat_imgs_pos'].to(self.device)  # [B, C, H, W]                                                                                                                                              │ │
                satimgs_neg = batch['sat_imgs_neg'].to(self.device)  # [B, C, H, W]                                                                                                                                              │ │
                coords_q = batch['coords_uav'].to(self.device)  # [B, 4]
                # debug for vis
                # q_id = 2
                # uav2vis = self.uav_datset_trian.denormalize_img(uavimgs[q_id])
                # satimg2vis = self.sat_dataset.denormalize_img(satimgs_pos[q_id])
                # from matplotlib import pyplot as plt
                # fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 一行两列
                # axes[0].imshow(uav2vis)
                # axes[1].imshow(satimg2vis)
                # plt.tight_layout()
                # plt.show()

                #get feat from vis_encoder
                imgs_input = torch.concatenate([uavimgs,satimgs_pos,satimgs_neg],dim=0)
                feats_input = self.vis_encoder(imgs_input.to(self.device))
                if self.agg_name == 'g2m':
                    feats_input = self.vis_aggregator(
                        feats_input[:, 1:, :].permute(0, 2, 1).reshape(feats_input.shape[0], self.feat_q_dim, 14,14))  # for patchsize=16,imgszie=224
                elif self.agg_name == 'salad':
                    feats_input = self.vis_aggregator(feats_input)  # aggregator = salad
                feats_q, feats_ref = feats_input[:uavimgs.shape[0]], feats_input[uavimgs.shape[0]:]
                feat_dist_mat = torch.norm(feats_q.unsqueeze(1)-feats_ref.unsqueeze(0),p=2,dim=-1)
                # feat_dist_mat_np = feat_dist_mat.detach().cpu().numpy()

                if not hasattr(self,'pos_mask_mat'):
                    self.pos_mask_mat = torch.concatenate([torch.eye(uavimgs.shape[0]),torch.zeros([feats_q.shape[0],satimgs_neg.shape[0]])],dim=-1).bool()
                loss = loss_swt(feat_dist_mat, self.pos_mask_mat)

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
                if it%10==0:
                    if self.writer is not None:
                        self.writer.add_scalar('loss_it', loss, step)
                    recall1 = (torch.argmin(feat_dist_mat, dim=-1) == torch.arange(0, feat_dist_mat.shape[0],
                                                                                   device=feat_dist_mat.device)).sum() / feat_dist_mat.shape[0]
                    self.logger.info(f'training set recall1={recall1.item():.4f}')
                step = step + 1

            # modify the learning rate
            # scheduler.step()

            # save the network's para
            if (epoch % 5 == 0) and (epoch>0): #for running
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
                # eval
                self.test_xy_scale_fm_vis_encoder()
                #log info:
                # grid_feats_grad = self.grid.codebook.feats.grad
                # self.logger.info(f"Grid feats grad L2 norm:{torch.linalg.norm(grid_feats_grad).item()}")
                # self.logger.info(f"Grid feats value.max():{feats_grid.max().item():.4f}")
                mean_dino = torch.mean(feats_q, dim=0)
                # mean_grid = torch.mean(feats_grid, dim=0)
                # l2_distance = torch.linalg.norm(mean_dino - mean_grid).item()
                # cosine_sim = F.cosine_similarity(mean_dino.unsqueeze(0), mean_grid.unsqueeze(0)).item()
                # self.logger.info(f"  mean值- L2 距离: {l2_distance:.6f}")
                # self.logger.info(f"  mean值- 余弦相似度: {cosine_sim:.6f}")
                # var_per_dim_grid = torch.var(feats_grid, dim=0)
                var_per_dim_dino = torch.var(feats_q, dim=0)
                # self.logger.info(f"  gird_方差均值: {var_per_dim_grid.mean().item():.6f}")
                self.logger.info(f"feat_方差的均值: {var_per_dim_dino.mean().item():.6f}")
                # 更有意义的指标：方差的方差（衡量特征维度的差异性）
                var_of_var = torch.var(var_per_dim_dino)
                self.logger.info(f"feat_方差的方差: {var_of_var.item():.6f}")

                if self.writer is not None:
                    # self.writer.add_scalar(' mean_L2_dist', l2_distance, epoch)
                    # self.writer.add_scalar(' mean_cosine_sim', cosine_sim, epoch)
                    # self.writer.add_scalar(' gird_var_mean', var_per_dim_grid.mean().item(), epoch)
                    self.writer.add_scalar(' feat_mean', mean_dino.mean().item(), epoch)
                    self.writer.add_scalar(' feat_var_mean', var_per_dim_dino.mean().item(), epoch)
                    self.writer.add_scalar(' feat_var_var', var_of_var.item(), epoch)

            # log info
            str2log = ''
            str2log += f'epcoh={epoch}'
            logger.info(f'loss={loss.item()}')
            time_elapsed = time.time() - since
            since = time.time()
            logger.info('epoch{:.0f} finished in {:.0f}m {:.0f}s'.format(epoch,time_elapsed // 60, time_elapsed % 6))
            logger.info('-' * 50)
            if self.writer is not None:
                self.writer.add_scalar('loss_epoch', loss, epoch)

            # backup the py info, at least have trained a epoch
            if epoch==0:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(exp_dir2save, self.opt)


    """"##############################
    functions awaiting internal invocation
    ##############################"""
    def _get_feats_fm_imgs(self, imgs_flatten, feat_fm_agg=True):
        batchsize = 512
        feat_galllery = []
        for batch in torch.split(imgs_flatten, batchsize, dim=0):
            feat = self.vis_encoder(batch.to(self.device))
            if feat_fm_agg:
                if self.agg_name == 'salad':
                    feat = self.vis_aggregator.forward(feat)
                elif self.agg_name == 'g2m':
                    feat = self.vis_aggregator.forward(feat[:, 1:, :].permute(0, 2, 1).reshape(feat.shape[0], self.feat_q_dim, 14, 14),normalize=True)
            else:
                feat = feat[:, 0, :]
            feat_galllery.append(feat.detach().cpu())
        feat_galllery = torch.cat(feat_galllery)
        return feat_galllery


    """"##############################
    functions about testing
    ##############################"""
    def test_xy_scale_fm_vis_encoder(self):
        # self._test_ready()
        overlap = 0.25

        # ==================== 生成尺度列表 ====================
        n_scales=3
        scale_list, satimgsize_list = self.sat_dataset.mk_sacle_levels(n_scales)
        print(f"\n尺度列表:")
        for i, (scale, imgsize) in enumerate(zip(scale_list, satimgsize_list)):
            print(f"  Level {i}: scale={scale:.3f}, imgsize={imgsize:.1f}px")

        # ==================== 构造特征库 ====================
        gallery_features = []  # 存储所有尺度的特征
        gallery_coords = []  # 存储对应的4D坐标
        gallery_shape = []
        for scale_idx, (scale, satimgsize2crop) in enumerate(zip(scale_list, satimgsize_list)):
            print(f"\n{'='*60}")
            print(f"处理尺度 {scale_idx+1}/{n_scales}: scale={scale:.3f}")
            print(f"{'='*60}")

            # ========== 1. 均匀裁剪卫星地图 ==========
            sat_tiles, nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
                size2clip=satimgsize2crop,
                overlap=overlap
            )
            n_rows, n_cols = sat_tiles.shape[:2]
            print(f"  裁剪网格大小: {n_rows} x {n_cols} = {n_rows*n_cols} 个位置")

            # resize到网络输入尺寸
            sat_tiles_resized = self.sat_dataset.scale_transform(sat_tiles.flatten(start_dim=0, end_dim=1))  # [n_pos, C, H, W]

            # ========== 3. 批量提取特征 ==========
            print(f"步骤3: 提取特征...")
            feats_all = self._get_feats_fm_imgs(sat_tiles_resized)

            # ========== 4. 构造4D坐标 ==========
            print(f"步骤4: 构造4D坐标...")
            nrcs_gallery_torch = torch.from_numpy(nrcs_gallery)  # [n_rows, n_cols, 2]
            rots_zero = torch.zeros((n_rows, n_cols, 1))  # [n_rows, n_cols, 1]
            scales_expanded = torch.full((n_rows, n_cols, 1), scale)  # [n_rows, n_cols, 1]
            coords_4d = torch.cat([
                nrcs_gallery_torch,  # [n_rows, n_cols, 2]
                rots_zero,  # [n_rows, n_cols, 1]
                scales_expanded  # [n_rows, n_cols, 1]
            ], dim=-1)
            # Flatten坐标: [n_rows * n_cols * n_rots, 4]
            coords_4d_flat = coords_4d.flatten(start_dim=0, end_dim=1)

            gallery_features.append(feats_all)
            gallery_coords.append(coords_4d_flat)
            gallery_shape.append(torch.Size([n_rows, n_cols]))

        feat_gallery_flatten_all = torch.concatenate(gallery_features,dim=0)
        coords_gallery_flatten_all = torch.concatenate(gallery_coords,dim=0)

        #sample query from dataloader
        uavimgs, coords_uav = next(iter(self.uav_dataloader_test))
        rot_to_align_deg = torch.rad2deg(-coords_uav[:,2]).cpu().numpy()  # 逆向旋转角度（度数）
        # 旋转UAV图像
        from util_batch_rotation import batch_rotate_images_per_sample
        uavimgs_wo_rot = batch_rotate_images_per_sample(
            uavimgs,  # [B, C, H, W]
            rot_to_align_deg  # [B] - 每个图像对应一个角度
        )  # 输出: [B, C, H, W]
        coords_uav_wo_rot = coords_uav.clone()
        coords_uav_wo_rot[:, 2] = 0  # rot = 0

        # ========== 采样对应的正样本卫星图（也是rot=0） ==========
        satimgs_pos = self.sat_dataset.crop_satimg_by_4d_coords(coords_uav_wo_rot)
        with torch.no_grad():
            feats_q = self._get_feats_fm_imgs(uavimgs_wo_rot)
            feats_pos = self._get_feats_fm_imgs(satimgs_pos)

        #eval
        import faiss
        topN=50
        feat_gallery_index_l2 = faiss.IndexFlatL2(self.feat_q_dim)
        feat_gallery_index_l2.add(feat_gallery_flatten_all.detach().cpu().numpy())
        feat_dist_l2, indices_l2 = feat_gallery_index_l2.search(feats_q.detach().cpu().numpy(), k=topN)

        coords_gallery_topN = coords_gallery_flatten_all[indices_l2[:,:topN]]
        dist_nrc_topN = torch.norm( coords_uav[:,None,:2].to(coords_gallery_topN.device)-coords_gallery_topN[:,:,:2],p=2, dim=-1)
        nrc_loc_success = dist_nrc_topN<self.sat_dataset.halfimg_radius_nrc*2
        k_values = [1,5,10,20,50]
        recalls = [(nrc_loc_success[:, :k].sum(dim=-1) > 0).float().mean().item() for k in k_values]
        info2log=f"Recall@K: " + " | ".join([f"R@{k}={r * 100:.3f}%" for k, r in zip(k_values, recalls)])
        self.logger.info(info2log)

        # 估检索到的 top1 特征与真实positive 特征的质量差异
        # feat_dist_pos2q = torch.norm(feats_pos-feats_q,dim=-1,p=2)
        # margin = feat_dist_pos2q - feat_dist_l2[:,0]
        # ratio = feat_dist_pos2q / feat_dist_l2[:,0]

        #debug
        # uav2vis = self.uav_datset_test.denormalize_img(uavimgs_wo_rot[10])
        # satimg2vis = self.sat_dataset.denormalize_img(satimgs_pos[10])
        # from matplotlib import pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 一行两列
        # axes[0].imshow(uav2vis)
        # axes[1].imshow(satimg2vis)
        # plt.tight_layout()
        # plt.show()


    def test_xy_rot_scale_fm_vis_encoder(self):
        # self._test_ready()
        overlap = 0.25

        # ==================== 生成尺度列表 ====================
        n_scales=3
        scale_list, satimgsize_list = self.sat_dataset.mk_sacle_levels(n_scales)
        print(f"\n尺度列表:")
        for i, (scale, imgsize) in enumerate(zip(scale_list, satimgsize_list)):
            print(f"  Level {i}: scale={scale:.3f}, imgsize={imgsize:.1f}px")

        # ==================== 准备旋转器 ====================
        delta_rot_rangle = 20
        rots_deg = [-180 + delta_rot_rangle * i for i in range(360 // delta_rot_rangle)]
        rots_rad = torch.tensor(np.deg2rad(np.stack(rots_deg)), dtype=torch.float32)
        n_rots = len(rots_rad)

        # ==================== 构造特征库 ====================
        gallery_features = []  # 存储所有尺度的特征
        gallery_coords = []  # 存储对应的4D坐标
        gallery_shape = []
        for scale_idx, (scale, satimgsize2crop) in enumerate(zip(scale_list, satimgsize_list)):
            print(f"\n{'='*60}")
            print(f"处理尺度 {scale_idx+1}/{n_scales}: scale={scale:.3f}")
            print(f"{'='*60}")

            # ========== 1. 均匀裁剪卫星地图 ==========
            sat_tiles, nrcs_gallery = self.sat_dataset.crop_sat_unifrom(
                size2clip=satimgsize2crop,
                overlap=overlap
            )
            n_rows, n_cols = sat_tiles.shape[:2]
            print(f"  裁剪网格大小: {n_rows} x {n_cols} = {n_rows*n_cols} 个位置")

            # Flatten位置维度
            sat_tiles_flatten = sat_tiles.flatten(start_dim=0, end_dim=1)  # [n_pos, C, H, W]
            # nrcs_gallery_flatten = torch.from_numpy(nrcs_gallery).flatten(start_dim=0, end_dim=1)  # [n_pos, 2]
            # n_positions = sat_tiles_flatten.shape[0]

            # ========== 2. 对每个位置旋转多个角度 ==========
            # 预处理：resize到网络输入尺寸
            sat_tiles_resized = self.sat_dataset.scale_transform(sat_tiles_flatten)  # [n_pos, C, H, W]

            # 旋转所有角度
            from util_batch_rotation import batch_rotate_images
            sat_tiles_rotated = batch_rotate_images(sat_tiles_resized,rots_deg)

            # ========== 3. 批量提取特征 ==========
            print(f"步骤3: 提取特征...")
            feats_all = self._get_feats_fm_imgs(sat_tiles_rotated.flatten(start_dim=0, end_dim=1))

            # ========== 4. 构造4D坐标 ==========
            print(f"步骤4: 构造4D坐标...")
            nrcs_gallery_torch = torch.from_numpy(nrcs_gallery)
            # 扩展nrcs: [n_rows, n_cols, 2] -> [n_rows, n_cols, n_rots, 2]
            nrcs_expanded = nrcs_gallery_torch[:, :, None, :].repeat(1, 1, n_rots, 1)
            # 扩展rots: [n_rots] -> [n_rows, n_cols, n_rots, 1]
            rots_expanded = rots_rad[None, None, :, None].repeat(n_rows, n_cols, 1, 1)
            # 扩展scales: scalar -> [n_rows, n_cols, n_rots, 1]
            scales_expanded = torch.full((n_rows, n_cols, n_rots, 1), scale)
            # 拼接: [n_rows, n_cols, n_rots, 4]
            coords_4d = torch.cat([
                nrcs_expanded,  # 已经是tensor
                rots_expanded,
                scales_expanded
            ], dim=-1)
            # Flatten坐标: [n_rows * n_cols * n_rots, 4]
            coords_4d_flat = coords_4d.flatten(start_dim=0, end_dim=2)

            gallery_features.append(feats_all)
            gallery_coords.append(coords_4d_flat)
            gallery_shape.append(torch.Size([n_rows, n_cols, n_rots]))

        feat_gallery_flatten_all = torch.concatenate(gallery_features,dim=0)
        coords_gallery_flatten_all = torch.concatenate(gallery_coords,dim=0)

        #sample query from dataloader
        uavimgs, coords_uav = next(iter(self.uav_dataloader_test))
        satimgs_pos = self.sat_dataset.crop_satimg_by_4d_coords(coords_uav)
        with torch.no_grad():
            feats_q = self._get_feats_fm_imgs(uavimgs)
            feats_pos = self._get_feats_fm_imgs(satimgs_pos)

        #eval
        import faiss
        topN=50
        feat_gallery_index_l2 = faiss.IndexFlatL2(self.feat_q_dim)
        feat_gallery_index_l2.add(feat_gallery_flatten_all.detach().cpu().numpy())
        feat_dist_l2, indices_l2 = feat_gallery_index_l2.search(feats_q.detach().cpu().numpy(), k=topN)

        coords_gallery_topN = coords_gallery_flatten_all[indices_l2[:,:topN]]
        dist_nrc_topN = torch.norm( coords_uav[:,None,:2].to(coords_gallery_topN.device)-coords_gallery_topN[:,:,:2],p=2, dim=-1)
        nrc_loc_success = dist_nrc_topN<self.sat_dataset.halfimg_radius_nrc*2
        k_values = [1,5,10,20,50]
        recalls = [(nrc_loc_success[:, :k].sum(dim=-1) > 0).float().mean().item() for k in k_values]
        print(f"Recall@K: " + " | ".join([f"R@{k}={r * 100:.3f}%" for k, r in zip(k_values, recalls)]))

        # 估检索到的 top1 特征与真实positive 特征的质量差异
        feat_dist_pos2q = torch.norm(feats_pos-feats_q,dim=-1,p=2)
        margin = feat_dist_pos2q - feat_dist_l2[:,0]
        ratio = feat_dist_pos2q / feat_dist_l2[:,0]

        # debug for vis
        # satimg2vis = self.sat_dataset.denormalize_img(satimgs_pos[1])
        # uav2vis = self.uav_datset_test.denormalize_img(uavimgs[1])
        # from matplotlib import pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 一行两列
        # axes[0].imshow(uav2vis)
        # axes[1].imshow(satimg2vis)
        # plt.tight_layout()
        # plt.show()


if __name__ == '__main__':
    torch.manual_seed(666)
    np.random.seed(2025)
    trainer = Trainer()
    trainer.train()
    # trainer.test_xy_rot()
    # trainer.mk_map_feats()
    # trainer.test_xy()
    # trainer.output_test_res()
    # trainer.test_rot()
    # trainer.test_radon_wo_translate()
    # trainer.test_radon_wo_translate_crossdomain()