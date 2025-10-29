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
import torch.nn.functional as F
import glob

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
        self.rc_pos_encoder = PositionalEncoder(input_dims=2,include_input=True,multires=6)
        self.rot_pos_encoder = PositionalEncoder(input_dims=2,include_input=True,multires=4)
        self.scale_pos_encoder = PositionalEncoder(input_dims=1,include_input=True,multires=4)
        coord_dim = self.rc_pos_encoder.out_dim+self.rot_pos_encoder.out_dim+self.scale_pos_encoder.out_dim
        from models.ocn_mlp import LocalDecoderFiLM,SerialModulator
        feat_q_dim = self.img_encoder.backbone.output_channel
        self.decoder = LocalDecoderFiLM(dim=feat_q_dim,c_dim=feat_q_dim,hidden_size=1024,n_blocks=3,output_dim=1,norm_type='none',leaky=True).to(self.device)

        # config the grid
        from app.nerf.main_nerf import NeRFAppConfig
        from wisp.config._tyro import parse_args_tyro_v1
        self.grid_args = parse_args_tyro_v1(NeRFAppConfig,'/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/nerf_hash.yaml')
        from wisp.config import instantiate
        blas = instantiate(self.grid_args.blas, pointcloud=None)
        self.grid = instantiate(self.grid_args.grid, blas=blas).to(self.device)  # A grid keeps track of both features and occupancy
        # self.grid_mlp = create_mlp([coord_dim+feat_q_dim,feat_q_dim,feat_q_dim],norm_type='layer').to(self.device)
        self.grid_mlp = SerialModulator(s_dim=feat_q_dim,c_dim=coord_dim+feat_q_dim, hidden_size=1024,n_blocks=5,output_dim=1024,c_operation='add',leaky=True).to(self.device)

        #define the param to save/laod
        self.param = {
            'grid':self.grid,
            'grid_mlp':self.grid_mlp,
        }

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
        for k,v in self.param.items():
            v.eval()
        self.img_encoder.eval()

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
        self.optimizer = create_optimizer_w_temple({"img_encoder":self.img_encoder,"gird_mlp":self.grid_mlp,"grid":self.grid},'adam')

        # load the ckpt for continuing train if necessray
        from tool.util_ckpt_handler import load_param
        if opt.load2train is not None and len(opt.load2train)>0:
            params_to_add = {"optimizer_state": self.optimizer}
            self.param.update(params_to_add)
            load_param(opt.load2train, self.param)
        begin_epoch = 0

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
        loss_mse = torch.nn.MSELoss(reduction='mean')
        # rc_radius = self.dataloader_train.dataset.sat_dataset.halfimg_radius_nrc * 0.5
        for epoch in range(begin_epoch,num_epochs):
            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            for it,data in tqdm.tqdm(enumerate(self.sat_dataloader)):
                satimg, sat_nrc, rad_roted, satimgsize_cover_ratio = data
                # uav_imgs, uav_nrcs = next(iter(self.uav_dataloader_train))
                coords = torch.concatenate([sat_nrc, rad_roted, satimgsize_cover_ratio], 1).to(self.device)

                #3.结果预测
                # get feats from hash
                feats_grid = self.feats_fm_grid(coords)
                from models.pos_encoder import encode_4d_coords
                coords_encoded = encode_4d_coords(torch.concatenate([sat_nrc,rad_roted,satimgsize_cover_ratio],dim=-1)
                                                         ,self.rc_pos_encoder,self.rot_pos_encoder,self.scale_pos_encoder).to(feats_grid.device)
                # feats_grid = self.grid_mlp(torch.concatenate([feats_grid,coords_encoded.to(feats_grid.device)],dim=-1))
                # feats_grid = self.grid_mlp(s=feats_grid,c=coords_encoded)
                feats_grid = self.grid_mlp(s=feats_grid, c=torch.concatenate([feats_grid,coords_encoded.to(feats_grid.device)],dim=-1))
                # get out put from mlp
                # from models.pos_encoder import encode_4d_coords
                # coords_sample_encoded = encode_4d_coords(coords, self.rc_pos_encoder, self.rot_pos_encoder, self.scale_pos_encoder)
                feats_q = self.img_encoder(satimg.to(self.device))
                # feats_q = feats_q.unsqueeze(1).expand(-1, coords_sample_encoded.shape[1], -1)

                # debug,方差分析：
                # var_per_dim_grid = torch.var(feats_grid, dim=0)
                # var_per_dim_dino = torch.var(feats_q, dim=0)
                # percentile = 0.95
                # threshold = torch.quantile(var_per_dim_dino, percentile)
                # #   使用阈值过滤两个方差分布
                # filtered_dino_vars = var_per_dim_dino[var_per_dim_dino < threshold]
                # filtered_grid_vars = var_per_dim_grid[var_per_dim_grid < threshold]
                # #   在同一张图上可视化比较两个过滤后的分布
                # plt.figure(figsize=(12, 7))
                # bins = 40
                # plot_range = (0, threshold.item())
                # plt.hist(filtered_dino_vars.detach().cpu().numpy(), bins=bins, range=plot_range,
                #          alpha=0.7, label='DINO Features (Filtered)', density=True)
                # plt.hist(filtered_grid_vars.detach().cpu().numpy(), bins=bins, range=plot_range,
                #          alpha=0.7, label='Hash Grid Features (Filtered)', density=True)
                # plt.title('Comparison of Filtered Variance Distributions (Top 99%)', fontsize=16)
                # plt.xlabel('Variance Value', fontsize=12)
                # plt.ylabel('Density', fontsize=12)
                # plt.legend(fontsize=12)
                # plt.grid(True, which="both", ls="--")
                # plt.show()
                # # # debug,均值分析：
                # mean_dino = torch.mean(feats_q, dim=0)
                # mean_grid = torch.mean(feats_grid, dim=0)
                # l2_distance = torch.linalg.norm(mean_dino - mean_grid).item()
                # cosine_sim = F.cosine_similarity(mean_dino.unsqueeze(0), mean_grid.unsqueeze(0)).item()
                # print(f"  - L2 距离: {l2_distance:.6f}")
                # print(f"  - 余弦相似度: {cosine_sim:.6f}")
                # plt.figure(figsize=(15, 7))
                # plt.plot(mean_dino.detach().cpu().numpy(), label='DINO Mean Feature', alpha=0.8)
                # plt.plot(mean_grid.detach().cpu().numpy(), label='Hash Grid Mean Feature', alpha=0.8, linestyle='--')
                # plt.title('Dimension-wise Comparison of Mean Feature Vectors', fontsize=16)
                # plt.xlabel('Feature Dimension Index', fontsize=12)
                # plt.ylabel('Mean Value', fontsize=12)
                # plt.legend(fontsize=12)
                # plt.grid(True, ls="--")
                # plt.show()
                # #均值可视化改进
                # plt.figure(figsize=(8, 8), dpi=100)  # 创建一个正方形的画布
                # ax = plt.gca()  # 获取当前的坐标轴
                # # 3. 绘制散点图
                # ax.scatter(mean_grid.detach().cpu().numpy(), mean_dino.detach().cpu().numpy(), alpha=0.4, s=15,
                #            label='Dimension-wise Values')
                # # 4. 绘制 y=x 对角线作为完美匹配的参考
                # min_val = min(mean_grid.min(), mean_dino.min()).item()
                # max_val = max(mean_grid.max(), mean_dino.max()).item()
                # ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2,
                #         label='y=x (Perfect Match)')
                # # 5. 设置图表样式和标签
                # ax.set_title('DINO vs. Hash Grid Feature Fit Analysis', fontsize=16)
                # ax.set_xlabel('Hash Grid Mean Feature Value', fontsize=12)
                # ax.set_ylabel('DINO Mean Feature Value', fontsize=12)
                # ax.legend(fontsize=12)
                # ax.grid(True, ls="--")
                # # 关键: 保持XY轴比例一致，确保 y=x 是准确的45度角
                # ax.set_aspect('equal', adjustable='box')
                # # 6. 在图上直接标注统计数据
                # stats_text = f'L2 Distance: {l2_distance:.4f}\nCosine Similarity: {cosine_sim:.4f}'
                # ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                #         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
                # plt.tight_layout()
                # plt.show()

                #4.loss构造
                # loss = torch.nn.functional.smooth_l1_loss(F.normalize(feats_grid.squeeze(),dim=-1),F.normalize(feats_q.squeeze(),dim=-1))
                # loss = torch.nn.functional.smooth_l1_loss(feats_grid.squeeze(),feats_q.squeeze())
                loss = loss_mse(feats_grid.squeeze(),feats_q.squeeze())
                # loss = torch.abs(feats_q.squeeze()-feats_grid.squeeze()).mean()

                # 反向传播
                self.optimizer.zero_grad()
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
                        self.writer.add_scalar('loss_it', loss, step)
                step = step + 1

            # modify the learning rate
            # scheduler.step()

            # save the network's para
            if epoch % 100 == 9: #for running
            # if epoch % 2 == 0: #for debugging
            # if (epoch == 10) or (epoch % 10 == 9 and epoch >= 110):
                # if ((epoch > 0) and (epoch % opt.save_freq == 0)) or ( epoch % 10 == 9 and epoch >= 110 ):
                from tool.util_ckpt_handler import save_param
                params_to_add = {
                    "optimizer_state": self.optimizer,
                    "epoch": epoch,
                }
                self.param.update(params_to_add)
                save_param(opt.exp_name, self.param)
                #log info:
                grid_feats_grad = self.grid.codebook.feats.grad
                self.logger.info(f"Grid feats grad L2 norm:{torch.linalg.norm(grid_feats_grad).item()}")
                self.logger.info(f"Grid feats value.max():{feats_grid.max().item():.4f}")
                mean_dino = torch.mean(feats_q, dim=0)
                mean_grid = torch.mean(feats_grid, dim=0)
                l2_distance = torch.linalg.norm(mean_dino - mean_grid).item()
                cosine_sim = F.cosine_similarity(mean_dino.unsqueeze(0), mean_grid.unsqueeze(0)).item()
                self.logger.info(f"  mean值- L2 距离: {l2_distance:.6f}")
                self.logger.info(f"  mean值- 余弦相似度: {cosine_sim:.6f}")
                var_per_dim_grid = torch.var(feats_grid, dim=0)
                var_per_dim_dino = torch.var(feats_q, dim=0)
                self.logger.info(f"  gird_方差均值: {var_per_dim_grid.mean().item():.6f}")
                self.logger.info(f"  dino_方差均值: {var_per_dim_dino.mean().item():.6f}")
                if self.writer is not None:
                    self.writer.add_scalar(' mean_L2_dist', l2_distance, epoch)
                    self.writer.add_scalar(' mean_cosine_sim', cosine_sim, epoch)
                    self.writer.add_scalar(' gird_var_mean', var_per_dim_grid.mean().item(), epoch)
                    self.writer.add_scalar(' dino_var_mean', var_per_dim_dino.mean().item(), epoch)

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

    def test(self):
        self._test_ready()
        from util_vis_4d_fields_in_2d import FieldViser2D
        self.viser_2d = FieldViser2D()
        from models.pos_encoder import encode_4d_coords

        with torch.no_grad():
            for it, data in tqdm.tqdm(enumerate(self.sat_dataloader)):
                satimg, sat_nrc, rad_roted, satimgsize_cover_ratio = data #the rot_rad are limited in [0,2pi]
                # uav_imgs, uav_nrcs = next(iter(self.uav_dataloader_train))
                gird_coords=self.viser_2d.mk_grid_nrcs(
                    row_range=self.sat_dataset.nr2sample_range,
                    col_range=self.sat_dataset.nc2sample_range,
                    resolution=64,
                    d_val=rad_roted[0].item(),
                    s_val=satimgsize_cover_ratio[0].item(),
                )
                feats_grid = self.feats_fm_grid(gird_coords.to(self.device))
                coords_encoded = encode_4d_coords(
                    torch.concatenate([sat_nrc, rad_roted, satimgsize_cover_ratio], dim=-1)
                    , self.rc_pos_encoder, self.rot_pos_encoder, self.scale_pos_encoder)
                # feats_grid = self.grid_mlp(torch.concatenate([feats_grid, coords_encoded.expand(feats_grid.shape[0],-1).to(feats_grid.device)], dim=-1))
                feats_grid = self.grid_mlp(s=feats_grid,c=torch.concatenate([feats_grid, coords_encoded.expand(feats_grid.shape[0],-1).to(feats_grid.device)], dim=-1))
                feats_q = self.img_encoder(satimg.to(self.device))
                # pred = F.normalize(feats_grid,dim=-1)@F.normalize(feats_q,dim=-1).T
                pred = torch.norm(feats_q.squeeze()-feats_grid.squeeze(),dim=-1,p=2)
                v,id = torch.sort(pred.squeeze(), dim=-1,descending=False)
                dist_v,dist_id = torch.sort(torch.norm(sat_nrc-gird_coords[:,:2],dim=-1), dim=-1,descending=False)
                # v[0]**2/1024=loss_mse,v[0]= the min norm
                # loss_mse = v[0]**2/feats_q.shape[-1]
                self.viser_2d.vis(pred.squeeze().detach().cpu().numpy(),gt_rc=sat_nrc.cpu().numpy(),extreme='min')

                # feats = self.img_encoder(satimg.to(self.device))
                # from util_vis_4d_fields_in_2d import visualize_udf_slice_rc
                # visualize_udf_slice_rc(
                #     row_range=self.sat_dataset.nr2sample_range,
                #     col_range=self.sat_dataset.nc2sample_range,
                #     d_val=rad_roted.item(),
                #     s_val=satimgsize_cover_ratio.item(),
                #     resolution=32,
                #     model_func=self.decoder,
                #     pos_encoders=[self.rc_pos_encoder,self.rot_pos_encoder,self.scale_pos_encoder],
                #     c_feat=feats[0],
                #     gt_rc=sat_nrc,
                # )

                # from util_vis_4d_fields_in_3d import visualize_udf_3d_rc_rot
                # visualize_udf_3d_rc_rot(
                #     r_range=self.sat_dataset.nr2sample_range,
                #     c_range=self.sat_dataset.nc2sample_range,
                #     rot_range=(0,2*math.pi),
                #     resolution=64,
                #     s_val=satimgsize_cover_ratio.item(),
                #     model_func=self.decoder,
                #     pos_encoders=[self.rc_pos_encoder,self.rot_pos_encoder,self.scale_pos_encoder],
                #     c_feat=feats[0],
                #     gt_pose=torch.concatenate([sat_nrc, rad_roted, satimgsize_cover_ratio],dim=-1).squeeze().numpy()
                # )

    def feats_fm_grid(self,coords):
        if len(coords.shape) == 3:
            n,m,_ = coords.shape
            coords_flatten = coords.flatten(start_dim=0, end_dim=1)
        else:
            coords_flatten = coords
        scales = coords_flatten[..., -1]

        grid_nrc_coords = coords_flatten[:, :2] * 2 - 1.
        grid_rot_coords = coords_flatten[:, 2:3] / (2 * torch.pi) - 1.
        grid_rot_coords *= (180/self.grid_args.grid.max_grid_res)
        gird_3d_coords = torch.concatenate([grid_nrc_coords, grid_rot_coords], dim=-1)
        n_gird_lod = len(self.grid.active_lods)
        feats_grid = self.grid.interpolate(gird_3d_coords.to(self.device), n_gird_lod - 1)
        #   aggerate the multiscale feats
        # normalized_scales = (scales - self.sat_dataset.satimgsize_scale_to_200m_boundary[0]) / (
        #             self.sat_dataset.satimgsize_scale_to_200m_boundary[1] -
        #             self.sat_dataset.satimgsize_scale_to_200m_boundary[0])
        normalized_scales = (self.sat_dataset.satimgsize_scale_to_200m_boundary[1] - scales) / (
                self.sat_dataset.satimgsize_scale_to_200m_boundary[1] - self.sat_dataset.satimgsize_scale_to_200m_boundary[0])
        center_indices = (normalized_scales * (n_gird_lod - 1)).squeeze()
        m_indices = torch.arange(n_gird_lod, device=normalized_scales.device)  # 形状: [M]

        dist = torch.abs(center_indices.unsqueeze(1) - m_indices.unsqueeze(0))
        epsilon = 1e-8
        # scale_weights = 1.0 / (dist + epsilon)
        p = 0.5  #p=2时，权重随距离二次方衰减（衰减更快）;p=0.5时，衰减更慢
        scale_weights = 1.0 / (dist.pow(p) + epsilon)
        scale_weights = scale_weights / torch.sum(scale_weights, dim=-1, keepdim=True)

        feats_grid = (feats_grid.reshape(gird_3d_coords.shape[0], n_gird_lod, -1) * scale_weights.unsqueeze(-1).to( self.device)).sum(dim=-2)

        if len(coords.shape) == 3:
            feats_grid = feats_grid.reshape(n, m, feats_grid.shape[-1])
        return feats_grid

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