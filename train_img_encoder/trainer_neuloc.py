# -*- coding: utf-8 -*-
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from __future__ import print_function, division
import argparse
import torch
import tqdm
from torch.cuda.amp import GradScaler
import time
from train_img_encoder.nets_taskflow import make_img_encoder
from tool.utils import get_logger, get_unique_exp_dir
# from tool.utils import load_network_wstate, save_network_wstate
import warnings
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import glob
import math

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
        self.img_encoder = make_img_encoder(self.opt).to(self.device)
            # freeze the para
        for param in self.img_encoder.parameters():
            param.requires_grad = False
        # config the mlp
        from models.pos_encoder import PositionalEncoder
        self.rc_pos_encoder = PositionalEncoder(input_dims=2,include_input=True,multires=10)
        self.rot_pos_encoder = PositionalEncoder(input_dims=2,include_input=True,multires=6)
        self.scale_pos_encoder = PositionalEncoder(input_dims=1,include_input=True,multires=6)
        coord_dim = self.rc_pos_encoder.out_dim+self.rot_pos_encoder.out_dim+self.scale_pos_encoder.out_dim
        from models.ocn_mlp import LocalDecoder
        self.decoder = LocalDecoder(dim=coord_dim,c_dim=self.img_encoder.backbone.output_channel,hidden_size=512,n_blocks=5,output_dim=1,c_opteration='mul',norm_type='none').to(self.device)

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
        para2load= {
            "img_encoder": self.img_encoder,
            "decoder": self.decoder,
        }
        load_param(opt.load2test,para2load)
        self.img_encoder.eval()
        self.decoder.eval()

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

        # load the ckpt for continuing train if necessray
        from tool.util_ckpt_handler import load_param
        if opt.load2train is not None and len(opt.load2train)>0:
            dict2load={
                "img_encoder": self.img_encoder,
                "decoder": self.decoder,
                "optimizer_state": self.optimizer,
                # "scheduler_state": self.lr_scheduler.state_dict(),
                "epoch": 0,
            }
            load_param(opt.load2train,dict2load)
            begin_epoch = dict2load["epoch"]
        else:
            begin_epoch = 0

        #config the optimizer ,todo:
        from tool.util_mk_optimizer import create_optimizer_w_temple
        self.optimizer = create_optimizer_w_temple({"img_encoder":self.img_encoder,"mlp":self.decoder},'adam')
        # self.optimizer, self.lr_scheduler = make_optimizer(self.img_encoder, self.opt)
        # optimizer,scheduler = self.optimizer,self.lr_scheduler

        # config the datalaoder:
        from dataset_wingtra_4d import SatDataset
        self.sat_dataset = SatDataset(
            p_satinfo_json=self.opt.p_satinfo_json,
            p_uav_geocsv=self.opt.p_uav_geocsv,
            imgsize2net=224,
        )
        self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset, batch_size = self.opt.batchsize_sat,
                                                          num_workers = self.opt.num_worker,
                                                          pin_memory=True, shuffle=True, drop_last=False,
                                                          persistent_workers=True)

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

        # ready to trian
        # self.val()
        self.img_encoder.train()  # Set model to training mode
        num_epochs = opt.num_epochs
        since = time.time()
        scaler = GradScaler()
        step = 0
        loss_mse = torch.nn.MSELoss(reduction='none')
        # rc_radius = self.dataloader_train.dataset.sat_dataset.halfimg_radius_nrc * 0.5
        for epoch in range(begin_epoch,num_epochs):
            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            for it,data in tqdm.tqdm(enumerate(self.sat_dataloader)):
                satimg, sat_nrc, rad_roted, satimgsize_cover_ratio = data
                # uav_imgs, uav_nrcs = next(iter(self.uav_dataloader_train))
                coords = torch.concatenate([sat_nrc, rad_roted, satimgsize_cover_ratio], 1).to(self.device)

                # making coords,verison1:
                from util_gen_coord_samples_hierarchical import generate_pose_samples_hierarchical,get_stratified_sampling_configs
                config_sampling = get_stratified_sampling_configs(
                    base_rc_std = self.sat_dataset.halfimg_radius_nrc,
                    base_dir_std_rad = np.deg2rad(10),
                    base_log_s_std = 0.05,
                )
                coords_sample  = generate_pose_samples_hierarchical(
                    p_true_batch=coords,
                    rc_bounds=(self.sat_dataset.nr2sample_range, self.sat_dataset.nc2sample_range),
                    scale_bounds=self.sat_dataset.satimgsize_scale_to_200m_boundary,
                    sampling_configs=config_sampling, # <--- 传入动态生成的配置
                    num_uniform_samples=512
                )
                # debug:vis the coord_samples
                # visualize_hierarchical_samples(
                #     p_true=coords[0],
                #     rc_bounds=(self.sat_dataset.nr2sample_range, self.sat_dataset.nc2sample_range),
                #     sampling_configs=config_sampling,
                #     all_samples=coords_sample[0],
                #     num_uniform_samples=512,
                # )

                # making coords,verison0:
                # from util_gen_coord_samples import generate_pose_samples
                # from models.pos_encoder import encode_4d_coords
                # coords_sample = generate_pose_samples(
                #     p_true_batch=coords,
                #     rc_bounds=(self.sat_dataset.nr2sample_range,self.sat_dataset.nc2sample_range),
                #     num_gaussian_samples=128,
                #     rc_std_dev=self.sat_dataset.halfimg_radius_nrc*0.5,
                #     direction_std_dev_rad=np.deg2rad(10),
                #     log_scale_std_dev=0.1,
                #     num_uniform_samples=1024,
                #     scale_bounds=self.sat_dataset.satimgsize_scale_to_200m_boundary,
                # )
                # #debug:vis the samples
                # from util_gen_coord_samples import visualize_samples
                # visualize_samples(
                #     samples_tensor=coords_sample[:1],
                #     anchors_tensor=coords[:1],
                #     rc_std_dev=self.sat_dataset.halfimg_radius_nrc,
                #     log_scale_std_dev=np.deg2rad(10),
                #     direction_std_dev_rad=0.1
                # )

                # making cooresponding distance
                diff_rcs = coords.unsqueeze(1)[:,:,:2] - coords_sample[:,:,:2]
                dists_rc = torch.norm(diff_rcs,dim=-1,keepdim=True)
                diff_rots = coords.unsqueeze(1)[:, :, 2:3] - coords_sample[:, :, 2:3]
                # 无论 diff_rots 的值有多大或多小，sin 和 cos 函数都会利用其周期性，将其映射到 [-1, 1] 的标准值域上。
                # 实际上，这一步是把一个“角度”转换为了单位圆上的一个点的 (x, y) 坐标，其中 x = cos(diff_rots)，y = sin(diff_rots)。
                # torch.atan2(y, x): atan2(y, x) 函数的作用是根据一个点的 (x, y) 坐标，计算出该点与原点连线和X轴正半轴之间的夹角。
                # 这个函数的一个重要特性是，它的输出值域被严格限制在 (-π, π] 之间（也就是-180度到+180度）
                dists_rot = torch.abs(torch.atan2(torch.sin(diff_rots), torch.cos(diff_rots)))
                diff_scales = coords.unsqueeze(1)[:, :,3:] /coords_sample[:, :,3:]
                dists_scale = torch.abs(torch.log(diff_scales)) #比例关系在对数空间中会变为加减关系

                norm_factor_rc = math.sqrt(self.sat_dataset.nr2sample_h**2+self.sat_dataset.nc2sample_w**2)
                nrom_factor_rot = torch.pi #todo:make the threshold auto
                nrom_factor_scale = math.log(self.sat_dataset.satimgsize_scale_to_200m_boundary[1]/self.sat_dataset.satimgsize_scale_to_200m_boundary[0])
                dists_rc_normed = dists_rc/norm_factor_rc
                dists_rot_normed = dists_rot/nrom_factor_rot
                dists_scale_normed = dists_scale/nrom_factor_scale

                # 1. 定义权重 (这是需要您根据实验调整的超参数)
                w_rc = 1.0  # 位置权重，通常设为1.0作为基准
                dist_threshold_accpetable = self.sat_dataset.halfimg_radius_nrc
                w_d = dist_threshold_accpetable/norm_factor_rc * 0.5# 方向权重
                w_s = w_d*0.5 # 尺度权重
                # 2. 计算加权的平方和
                dist_true_squared = (
                        w_rc * (dists_rc_normed ** 2) +
                        w_d * (dists_rot_normed ** 2) +
                        w_s * (dists_scale_normed ** 2)
                )
                # 3. (可选但推荐) 取平方根，得到最终的距离
                # 这使得 dist_true 的“单位”与 dist_pred 保持一致，损失函数更稳定
                dist_label = torch.sqrt(dist_true_squared) + 1e-7

                # making feats
                from models.pos_encoder import encode_4d_coords
                coords_sample_encoded = encode_4d_coords(coords_sample,self.rc_pos_encoder,self.rot_pos_encoder,self.scale_pos_encoder)
                feats = self.img_encoder(satimg.to(self.device))
                feats = feats.unsqueeze(1).expand(-1, coords_sample_encoded.shape[1], -1)

                # pred_dist
                dist_pred = self.decoder(coords_sample_encoded, feats)
                # dist_pred = torch.relu(dist_pred)

                # 3. 创建权重张量 (weights tensor)
                sigma = self.sat_dataset.halfimg_radius_nrc*2  # 您提到的“UDF距离阈值”，这是关键的调节参数！
                base_weight = 1.0
                bonus_weight = 2.0  # 这使得在 dist_label=0 处的总权重为 1.0 + 9.0 = 10.0
                gaussian_component = torch.exp(-dist_label.squeeze().pow(2) / (2 * sigma ** 2))
                weights = base_weight + bonus_weight * gaussian_component
                # loss:
                # loss = torch.nn.functional.smooth_l1_loss(dist_pred.squeeze(),dist_label.squeeze())
                # loss = torch.abs(dist_pred.squeeze()-dist_label.squeeze()).sum(dim=-1).mean()
                loss = loss_mse(dist_pred.squeeze(), dist_label.squeeze())
                loss = loss * weights
                loss = loss.mean()

                #小批量迭代
                # coords_sample_encoded = coords_sample_encoded.flatten(start_dim=0,end_dim=1)
                # feats = feats.flatten(start_dim=0,end_dim=1)
                # dist_label = dist_label.flatten(start_dim=0,end_dim=1)
                # mini_batch_size = 4096  # 可以根据显存调整
                # num_iterations_per_batch = num_total_samples // mini_batch_size
                # # 4. 进入内部迭代循环
                # for i in range(num_iterations_per_batch):
                #     # a. 随机采样索引
                #     indices = torch.randperm(num_total_samples, device=self.device)[:mini_batch_size]
                #
                #     # b. 根据索引创建 mini_batch
                #     mini_coords = all_coords_encoded[indices]
                #     mini_feats = all_feats[indices]
                #     mini_labels = all_labels[indices]
                #
                #     # c. 梯度清零，前向传播，计算loss (和之前一样)
                #     self.optimizer.zero_grad()


                # 梯度清零
                self.optimizer.zero_grad()
                # if opt.autocast:
                #     with autocast():
                #         # making feats
                #         feats = self.img_encoder(satimg.to(self.device))
                #         feats = feats.unsqueeze(1).expand(-1, coords_sample_encoded.shape[1], -1)
                #         # pred dist
                #         dist_pred = self.decoder(coords_sample_encoded, feats)
                # else:
                #     # making feats
                #     feats = self.img_encoder(satimg.to(self.device))
                #     feats = feats.unsqueeze(1).expand(-1, coords_sample_encoded.shape[1], -1)
                #     # pred_dist
                #     dist_pred = self.decoder(coords_sample_encoded, feats)
                # # loss = torch.nn.functional.smooth_l1_loss(dist_pred.squeeze(),dist_label.squeeze())
                # # loss = torch.abs(dist_pred.squeeze()-dist_label.squeeze()).sum(dim=-1).mean()
                # loss = loss_mse(dist_pred.squeeze(), dist_label.squeeze())

                # 反向传播
                if opt.autocast:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # 输出loss指标
                if it%10==0:
                    if self.writer is not None:
                        self.writer.add_scalar('loss', loss, step)
                step += 1

            # modify the learning rate
            # scheduler.step()

            # save the network's para
            if epoch % 10 == 9:
            # if epoch % 2 == 0:
            # if (epoch == 10) or (epoch % 10 == 9 and epoch >= 110):
                # if ((epoch > 0) and (epoch % opt.save_freq == 0)) or ( epoch % 10 == 9 and epoch >= 110 ):
                from tool.util_ckpt_handler import save_param
                para2save = {
                    "img_encoder":self.img_encoder.state_dict(),
                    "decoder":self.decoder.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    # "scheduler_state": self.lr_scheduler.state_dict(),
                    "epoch": epoch,
                           }
                save_param(opt.exp_name, para2save)
                self.img_encoder.to(self.device)
                self.decoder.to(self.device)

            # val
            # self.val() if opt.val and (epoch % opt.val_freq ==0) else None
            self.img_encoder.train()

            # log info
            str2log = ''
            str2log += f'epcoh={epoch}'
            logger.info(f'loss={loss.item()}')
            time_elapsed = time.time() - since
            since = time.time()
            logger.info('epoch{:.0f} finished in {:.0f}m {:.0f}s'.format(epoch,time_elapsed // 60, time_elapsed % 6))
            logger.info('-' * 50)
            if self.writer is not None:
                self.writer.add_scalar('loss', loss, epoch)

            # backup the py info, at least have trained a epoch
            if epoch==0:
                from tool.util_backup_exp_by_git import backup_experiment
                backup_experiment(exp_dir2save, self.opt)

    def test(self):
        self._test_ready()

        for it, data in tqdm.tqdm(enumerate(self.sat_dataloader)):
            satimg, sat_nrc, rad_roted, satimgsize_cover_ratio = data #the rot_rad are limited in [0,2pi]
            # uav_imgs, uav_nrcs = next(iter(self.uav_dataloader_train))

            feats = self.img_encoder(satimg.to(self.device))
            from util_vis_4d_fields_in_2d import visualize_udf_slice_rc
            visualize_udf_slice_rc(
                row_range=self.sat_dataset.nr2sample_range,
                col_range=self.sat_dataset.nc2sample_range,
                d_val=rad_roted.item(),
                s_val=satimgsize_cover_ratio.item(),
                resolution=32,
                model_func=self.decoder,
                pos_encoders=[self.rc_pos_encoder,self.rot_pos_encoder,self.scale_pos_encoder],
                c_feat=feats[0],
                gt_rc=sat_nrc,
            )

            from util_vis_4d_fields_in_3d import visualize_udf_3d_rc_rot
            visualize_udf_3d_rc_rot(
                r_range=self.sat_dataset.nr2sample_range,
                c_range=self.sat_dataset.nc2sample_range,
                rot_range=(0,2*math.pi),
                resolution=64,
                s_val=satimgsize_cover_ratio.item(),
                model_func=self.decoder,
                pos_encoders=[self.rc_pos_encoder,self.rot_pos_encoder,self.scale_pos_encoder],
                c_feat=feats[0],
                gt_pose=torch.concatenate([sat_nrc, rad_roted, satimgsize_cover_ratio],dim=-1).squeeze().numpy()
            )


        # # get feat_refs from satimg
        # # --- 1. 装载所有瓦片至cpu ---
        # # 确保 crop_sat_unifrom 返回的是CPU上的Tensor或Numpy数组
        # sat_tiles, rc_gallery = self.dataloader_test.dataset.sat_dataset.crop_sat_unifrom(overlap=overlap)
        # # 在CPU上进行变形和缓存
        # sat_tiles = sat_tiles.reshape(-1, *sat_tiles.shape[2:])
        # print(f"已在CPU上缓存 {sat_tiles.shape[0]} 个瓦片。")
        # # --- 2. 分批处理瓦片，每次只将一小批数据移至GPU ---
        # split_size = 32  # 这是您送入模型的批大小 (batch size)
        # feat_gallery = []
        # sg_gallery = []
        # print("开始分批提取特征...")
        # for tiles_batch_cpu in tqdm.tqdm(torch.split(sat_tiles, split_size, dim=0), desc="Extracting Features"):
        #     tiles_batch_gpu = tiles_batch_cpu.to(self.device)
        #     with torch.no_grad():
        #         if with_sg:
        #             output, output_sgs = img_encoder(tiles_batch_gpu, ret_sg=True)
        #             sg_gallery.append(output_sgs)
        #         else:
        #             output = img_encoder(tiles_batch_gpu, ret_sg=False)
        #     output = output[1] if isinstance(output, list) else output
        #     # 将计算结果立即移回CPU，以释放GPU显存
        #     feat_g = output.detach().cpu()
        #     feat_g = feat_g.view(feat_g.shape[0], -1) if len(feat_g.shape) > 2 else feat_g
        #     feat_gallery.append(feat_g)
        # # --- 3. 在CPU上拼接所有批次的特征 ---
        # feat_gallery = torch.cat(feat_gallery, dim=0)
        # print("所有瓦片的特征已提取完成。")
        #
        # # get feat_querys from uavimgs
        # feat_query,rc_query = [],[]
        # sg_query = []
        # for data in tqdm.tqdm(dataloader):
        #     uavimg_q, uav_rc = data[0].to(self.device),data[1]
        #     if with_sg:
        #         output,output_sgs = img_encoder(uavimg_q,ret_sg=True)
        #         sg_gallery.append(output_sgs)
        #     else:
        #         output = img_encoder(uavimg_q, ret_sg=False)
        #     feat_q = output[1] if type(output)==list else output
        #     feat_q =  feat_q.view(feat_q.shape[0],-1) if len(feat_q.shape)>2 else feat_q
        #     feat_query.append(feat_q.detach().cpu())
        #     rc_query.append(uav_rc)
        #     sg_query.append(output_sgs) if with_sg else None
        # feat_query = torch.cat(feat_query,dim=0)
        # rc_query = np.concatenate(rc_query,axis=0)
        #
        # # get georc_info
        # georc_gallary = dataloader.dataset.sat_dataset.transform_nrc_to_georc(rc_gallery.reshape(-1,2)) #todo:
        # sg_gallery = torch.cat(sg_gallery, dim=0).detach().cpu() if with_sg else None
        # georc_query = dataloader.dataset.sat_dataset.transform_nrc_to_georc(rc_query)
        # sg_query = torch.cat(sg_query, dim=0).detach().cpu() if with_sg else None
        #
        # # get gt_label fro perd
        # from eval_recall_fm_salad import qurey_label_fm_gallery_nrc
        # distrc_sats2uav, uav_labels_query = qurey_label_fm_gallery_nrc(rc_gallery.reshape(-1, 2), rc_query, dataloader.dataset.sat_dataset.halfimg_radius_nrc)
        # rotdeg_fm_north_anticlock = dataloader.uav_dataset.rotdeg_fm_north_anticlock if hasattr(dataloader.dataset.uav_dataset,'rotdeg_fm_north_anticlock') else None
        #
        # # log the recall
        # from eval_recall_fm_salad import compute_recall_from_feat
        # d = compute_recall_from_feat(feat_query.contiguous(), feat_gallery.contiguous(), uav_labels_query,[1, 5, 20, 50, 200], faiss_gpu=False)
        # info = "Recall"
        # for k, v in d.items():
        #     info = info + f" @{k}:{v * 100:.2f} "
        # self.logger.info(info) if hasattr(self, "logger") else None
        # # write the recall to a txt
        # p_txt2write = os.path.join(os.path.dirname(self.opt.load2test), f"recall_overlap{overlap}.txt")
        # info2wirte = 'p_json=' + self.opt.p_satinfo_json
        # info2wirte += f'\nn_refs={feat_gallery.shape[0]}'
        # info2wirte += f'\nsatimgsize2crop={self.opt.satimgsize2crop}'
        # info2wirte += f'\nrecall_m={dataloader.dataset.sat_dataset.halfimg_radius_meter}'
        # # info2wirte += f'\nsample_overlap={overlap}'
        # info2wirte += info
        # with open(p_txt2write, 'a', encoding='utf-8') as f:
        #     f.write(info2wirte)
        #
        # if save_res:
        #     # save the result matrix:
        #     result = {'gallery_feat': feat_gallery.detach().cpu().numpy(),
        #               'gallery_rc': rc_gallery,
        #               'gallery_georc': georc_gallary,
        #               'gallery_hw':np.array(rc_gallery.shape[:2]),
        #               'query_feat': feat_query.detach().cpu().numpy(),
        #               'query_rc': rc_query,
        #               'query_latlon': georc_query,
        #               'query_label': uav_labels_query,
        #               'query_dist': distrc_sats2uav,
        #               'radius_rc': dataloader.dataset.sat_dataset.halfimg_radius_nrc,
        #               'radius_meter': dataloader.dataset.sat_dataset.halfimg_radius_meter,
        #               }
        #     if with_sg:
        #         result['gallery_sg'] = sg_gallery
        #         result['query_sg'] = sg_query
        #         result['rotdeg_fm_north_anticlock'] = rotdeg_fm_north_anticlock
        #
        #     suffix = f"_overlap{overlap}_radius{dataloader.dataset.sat_dataset.halfimg_radius_meter:.0f}m.mat"
        #     scipy.io.savemat( f"{self.opt.exps_dir}/{self.opt.exp_name}/{os.path.basename(self.opt.load2test).replace('.pth', suffix)}" ,result)



if __name__ == '__main__':
    torch.manual_seed(666)
    np.random.seed(2025)
    trainer = Trainer()
    # trainer.train()
    # trainer.val()
    trainer.test()
    # trainer.test_xy()
    # trainer.output_test_res()
    # trainer.test_rot()
    # trainer.test_radon_wo_translate()
    # trainer.test_radon_wo_translate_crossdomain()