# -*- coding: utf-8 -*-
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from __future__ import print_function, division
import argparse
import torch
import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import time
import scipy.io
from optimizers.make_optimizer import make_optimizer
from train_img_encoder.nets_taskflow import make_model
from tool.utils import copyfiles2checkpoints, get_logger, get_unique_exp_dir
from tool.utils import load_network_wstate, save_network_wstate
import warnings
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import glob

from losses.loss_cl import Loss
warnings.filterwarnings("ignore")

# var to selct:
# from datasets.make_dataloader import make_dataloader
# from datasets_custom.make_dataloder_classify import make_dataloader_train
from datasets_custom.make_dataloader_wingtra import make_dataloader_wingtra
# from exps.exp24.datasets_custom.make_dataloader_dsalad import  make_dataloader
from PIL import Image
from matplotlib import pyplot as plt
import yaml
import os
import scipy
import json


def json_dict(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON format for dictionary.")


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    #about exp setting
    parser.add_argument('--exp_dir', default='.exps/',type=str, help='the dir that save experiments')
    parser.add_argument('--exp_name', default='debug',type=str, help='the experiment name that will be saved in exps dir in the root')
    parser.add_argument('--p_yaml', default='/home/data/zwk/pyproj_DUAV_salad_6.4/opts_wingtra.yaml', type=str, help='the yaml file about the defult setting')
    parser.add_argument('--p_satinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu__512h_lineClassed/dataset_xmu_meta/satimg_xiangan_xmu_info.json',
                        type=str, help='training dir path')
    parser.add_argument('--p_uavinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu_512h_lineClassed/dataset_xmu_meta/uavimgs_xiangan_xmu_info.json',
                        type=str, help='training dir path')
    parser.add_argument('--dataset_name',default='xmu', type=str)
    parser.add_argument('--load2test', default="/home/data/zwk/pyproj_DUAV_salad_6.4/exps/exp_wohead_vit-b/epoch000.pth", type=str, help='path for testing') # for testing
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
    parser.add_argument('--batchsize', default = 32, type=int, help='batchsize')
    parser.add_argument('--autocast', action='store_true', default=True, help='use mix precision')
    #about data setting,version 2:
    parser.add_argument('--imgsize2net', default=224, type=int)
    parser.add_argument('--satimgsize2crop', default=224, type=int)
    parser.add_argument('--n_rand2sample_per_pos', default=2, type=int)
    parser.add_argument('--uav_da', nargs='+', default=['rr'],help='rr=random_rotate,ra=random affine,re=random erasing,cj=color jitter,cda=color data argument')
    parser.add_argument('--sat_da', nargs='+', default=['ra','re'],help='rr=random_rotate,ra=random affine,re=random erasing,cj=color jitter,cda=color data argument')
    parser.add_argument('--erasing_p', default=0.3, type=float,help='random erasing probability, in [0,1]')
    #about networks
    parser.add_argument('--w_classify', default=False, action='store_true', help='')
    parser.add_argument('--cls_loss', default="CELoss", type=str, help='loss type of representation learning')
    parser.add_argument('--kl_loss', default="KLLoss", type=str, help='loss type of mutual learning')
    parser.add_argument('--feature_loss', nargs='+', default=["WeightedSoftTripletLoss"],
                        help='"InfoNceLoss","MSLoss","TripletLoss","HardMiningTripletLoss","SameDomainTripletLoss","WeightedSoftTripletLoss","ContrastiveLoss"')
    parser.add_argument('--backbone', default="ViTB-384", type=str, help='ViTB-224;ViTS-224;dinov2_vitb14;ViTB-384')
    parser.add_argument('--head', default="", type=str, help='salad;FSRA;LPN;') #"" means no head
    parser.add_argument('--block', default=2, type=int, help='') #will by used when headF=FSRA,LPN,NetVLAD,NeXtVLAD
    parser.add_argument('--num_bottleneck', default=512, type=int, help='the dimensions for embedding the feature')
    parser.add_argument('--head_pool', default="avg", type=str, help='head pooling type for applying') #will by used when head=SingleBranch
    parser.add_argument('--wcls_token', default=False, type=bool) #will by used when head=SingleBranch
    parser.add_argument('--norm_output', default=True, type=bool)
    #about learning setting
    parser.add_argument('--num_epochs', default=50, type=int, help='total epoches for training')
    parser.add_argument('--warm_epoch', default=0, type=int,
                        help='the first K epoch that needs warm up')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
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
        'exp_setting': ['p_yaml', 'p_satinfo_json', 'p_uavinfo_json','exp_name','exp_dir','load2train', 'load2test', 'val','val_freq', 'save_freq', 'tensorboard'],
        'hardware_setting': ['gpu_ids', 'num_worker', 'batchsize', 'autocast'],
        'data_setting': ['imgsize2net', 'satimgsize2crop', 'n_rand2sample_per_pos', 'uav_da', 'sat_da', 'erasing_p'],
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


    def train(self):
        opt = self.opt
        self.model = make_model(self.opt)
        if self.opt.use_gpu:
            self.model = self.model.to(self.device)
        model = self.model
        self.optimizer_ft, self.exp_lr_scheduler = make_optimizer(self.model, self.opt)
        optimizer,scheduler = self.optimizer_ft,self.exp_lr_scheduler

        # config dataloader
        # self.dataloader = make_dataloader_gta(self.opt,stage='train') if self.opt.dataset_name == 'gta' else make_dataloader_xmu(self.opt,stage='train')
        # dataloader = self.dataloader
        self.dataloader_train,self.dataloader_test = make_dataloader_wingtra(opt)

        # load the ckpt for continuing train if necessray
        if opt.load2train is not None and len(opt.load2train)>0:
            begin_epoch = load_network_wstate(opt.load2train, model, optimizer, scheduler)
        else:
            begin_epoch = 0

        # config logger and backup files
        exp_name = get_unique_exp_dir(opt.exp_dir,opt.exp_name)
        opt.exp_name = exp_name
        os.makedirs(os.path.join(opt.exp_dir,opt.exp_name),exist_ok=True)
        logger = get_logger("{}/{}/train.log".format(opt.exp_dir,opt.exp_name),'trainer_logger')
        self.logger = logger
        self.logger.info(f"exp ready!, exp_name={exp_name}")
        copyfiles2checkpoints( self.opt )

        # config tensorborad if necessary
        writer = SummaryWriter("exps/{}/train_tensorboard.log".format(opt.exp_name)) if opt.tensorboard else None

        # config loss
        loss_dtype = torch.float16 if opt.autocast and opt.head!="" else torch.float32
        opt.loss_dtype = loss_dtype
        nnloss = Loss(opt)

        # ready to trian
        # self.val()
        num_epochs = opt.num_epochs
        since = time.time()
        scaler = GradScaler()
        step = 0
        rc_radius = self.dataloader_train.dataset.sat_dataset.halfimg_radius_nrc * 0.5
        for epoch in range(begin_epoch,num_epochs):
            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

            model.train(True)  # Set model to training mode

            for it,data in tqdm.tqdm(enumerate(self.dataloader_train)):
                # time_it10 = time.time()

                # 获取输入无人机和卫星数据
                imgs_d, imgs_s, imgs_s_rand, rcs_d, rcs_s, rcs_rand = data
                # debug
                # rcs_d, rcs_s, rcs_rand = None,None,None
                # imgs_d, imgs_s, imgs_s_rand, rcs_d, rcs_rand = data

                imgs = torch.concatenate([imgs_d,imgs_s,imgs_s_rand],dim=0).to(self.device)
                # rcs = torch.concatenate([rcs_pos,rcs_pos,rcs_rand],dim=0).to(self.device)
                n_d = imgs_d.shape[0]

                # 梯度清零
                optimizer.zero_grad()
                with autocast():
                    if opt.wcls_token and opt.head!="":
                        outputs, clses = model(imgs)
                    else:
                        outputs = model(imgs)
                outputs = torch.concatenate(outputs,dim=-1) if type(outputs) == list else outputs

                outputs = F.normalize(outputs,dim=-1) if opt.norm_output else outputs
                # loss_dict = nnloss.forward(outputs[:n_d],outputs[n_d:2*n_d],outputs[2*n_d:])
                loss_dict = nnloss.forward(outputs[:n_d], outputs[n_d:2 * n_d], outputs[2 * n_d:],rcs_d,rcs_s,rcs_rand,rc_radius) #query,pos,random

                # time_netbackwad_begin = time.time()
                # 反向传播
                if opt.autocast:
                    scaler.scale(loss_dict['all']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict['all'].backward()
                    optimizer.step()

                # 输出loss指标
                if it%10==0:
                    print(f"loss_all={loss_dict['all'].item():.4f}")
                    # time_it10 = time.time() - time_it10
                    # print(f"time_it10={time_it10:.6f}")
                    if opt.tensorboard:
                        writer.add_scalars('losses', loss_dict, step)
                step += 1

            scheduler.step()  # modify the learning rate
            # if epoch % 10 == 9 and epoch >= 110:
            # if epoch % 2 == 0:
            # if (epoch == 10) or (epoch % 10 == 9 and epoch >= 110):
                # if ((epoch > 0) and (epoch % opt.save_freq == 0)) or ( epoch % 10 == 9 and epoch >= 110 ):
            save_network_wstate(opt.exp_name, model, optimizer, scheduler, epoch)

            self.val() if opt.val and (epoch % opt.val_freq ==0) else None
            self.model.train()
            str2log = ''
            for k, v in loss_dict.items():
                str2log += f'{k}={v.item():.3f}; '
            str2log += f'epcoh={epoch}'
            logger.info(str2log)
            # if opt.tensorboard:
            #     writer.add_scalars('losses', loss_dict, epoch)
            time_elapsed = time.time() - since
            since = time.time()
            logger.info('epoch{:.0f} finished in {:.0f}m {:.0f}s'.format(epoch,time_elapsed // 60, time_elapsed % 6))
            logger.info('-' * 50)


    def val(self,overlap=0.25):
        torch.cuda.empty_cache()
        self.model.eval()
        from datasets_custom.eval_gta import GTAEvaluator
        if self.opt.dataset_name == 'gta':
            if not hasattr(self, 'gta_evaluator'):
                self.gta_evalator = GTAEvaluator(uav_transform=self.dataloader.dataset.uav_transform_test,sat_transform=self.dataloader.dataset.sat_transform_test)
            self.gta_evalator.eval_gta(model=self.model,logger=self.logger)
        else:
            with torch.no_grad():
                ###############################
                # --- 1. 在CPU上准备和缓存所有瓦片数据 ---
                # 这个逻辑确保 self.sat_tiles 是一个存储在CPU内存中的、巨大的瓦片张量
                resize_transform = torchvision.transforms.Resize(self.opt.imgsize2net)
                if not hasattr(self, 'sat_tiles') or self.sat_tiles is None:
                    print("首次运行，正在从数据集中裁剪瓦片...")
                    # 确保 crop_sat_unifrom 返回的是CPU上的Tensor或Numpy数组
                    sat_tiles_cpu, rc_gallery = self.dataloader_test.dataset.sat_dataset.crop_sat_unifrom(overlap=overlap,size2clip=self.opt.satimgsize2crop)
                    sat_tiles_cpu = resize_transform(sat_tiles_cpu.reshape(-1, *sat_tiles_cpu.shape[2:]))
                    # 在CPU上进行变形和缓存
                    self.sat_tiles = sat_tiles_cpu
                    self.rc_gallery = rc_gallery
                    print(f"已在CPU上缓存 {self.sat_tiles.shape[0]} 个瓦片。")
                else:
                    rc_gallery = self.rc_gallery
                    print("检测到已缓存的瓦片，直接从CPU内存使用。")

                # --- 2. 分批处理瓦片，每次只将一小批数据移至GPU ---
                split_size = 32  # 这是您送入模型的批大小 (batch size)
                feat_gallery = []
                print("开始分批提取特征...")
                # torch.split 会从巨大的CPU张量 self.sat_tiles 中切分出一小块 tiles_batch_cpu
                for tiles_batch_cpu in tqdm.tqdm(torch.split(self.sat_tiles, split_size, dim=0),desc="Extracting Features"):
                    # 关键改动：只将当前这一小批数据移动到GPU
                    tiles_batch_gpu = tiles_batch_cpu.to(self.device)
                    # 使用 torch.no_grad() 进行推理，可以节省显存并加速
                    with torch.no_grad():
                        # 将GPU上的批次送入模型
                        output = self.model(tiles_batch_gpu, ret_sg=False)
                    # (您的后续处理逻辑保持不变)
                    output = output[1] if isinstance(output, list) else output
                    # 将计算结果立即移回CPU，以释放GPU显存
                    feat_g = output.detach().cpu()
                    feat_g = feat_g.view(feat_g.shape[0], -1) if len(feat_g.shape) > 2 else feat_g
                    feat_gallery.append(feat_g)
                # --- 3. 在CPU上拼接所有批次的特征 ---
                feat_gallery = torch.cat(feat_gallery, dim=0)
                print("所有瓦片的特征已提取完成。")

                # get feat_querys from uavimgs
                feat_query,rc_query = [],[]
                for data in tqdm.tqdm(self.dataloader_test):
                    uavimg_q, uav_rc = data[0],data[1]
                    output = self.model(uavimg_q.to(self.device),ret_sg=False)
                    feat_q = output[1] if type(output)==list else output
                    feat_q =  feat_q.view(feat_q.shape[0],-1) if len(feat_q.shape)>2 else feat_q
                    feat_query.append(feat_q.detach().cpu())
                    rc_query.append(uav_rc)
                feat_query = torch.cat(feat_query,dim=0)
                rc_query = np.concatenate(rc_query,axis=0)

                from eval_recall_fm_salad import qurey_label_fm_gallery_nrc
                distrc_sats2uav,uav_labels_query = qurey_label_fm_gallery_nrc(rc_gallery.reshape(-1,2), rc_query, self.dataloader_test.dataset.sat_dataset.halfimg_radius_nrc)

                from eval_recall_fm_salad import compute_recall_from_feat
                d = compute_recall_from_feat(feat_query.contiguous(), feat_gallery.contiguous(), uav_labels_query, [1, 5, 20, 50, 200], faiss_gpu=False)
                info = "Recall"
                for k, v in d.items():
                    info = info + f" @{k}:{v * 100:.2f} "
                self.logger.info(info)

            self.model.train()


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

        self.model = make_model(self.opt)
        if self.opt.use_gpu:
            self.mdoel = self.model.to(self.device)
        # model.load_state_dict(torch.load(opt.checkpoint)) #org version
        checkpoint = torch.load(opt.load2test)
        self.model.load_state_dict(checkpoint["model_state"]) if "model_state" in checkpoint else self.mdoel.load_state_dict(checkpoint)
        self.model.eval()
        self.dataloader_train,self.dataloader_test = make_dataloader_wingtra(opt)
        return self.model, self.dataloader_test


    def test(self,overlap=0.5,with_sg=False,save_res=False ):
        model, dataloader = self._test_ready()

        # get feat_refs from satimg
        # --- 1. 装载所有瓦片至cpu ---
        # 确保 crop_sat_unifrom 返回的是CPU上的Tensor或Numpy数组
        sat_tiles, rc_gallery = self.dataloader_test.dataset.sat_dataset.crop_sat_unifrom(overlap=overlap)
        # 在CPU上进行变形和缓存
        sat_tiles = sat_tiles.reshape(-1, *sat_tiles.shape[2:])
        print(f"已在CPU上缓存 {sat_tiles.shape[0]} 个瓦片。")
        # --- 2. 分批处理瓦片，每次只将一小批数据移至GPU ---
        split_size = 32  # 这是您送入模型的批大小 (batch size)
        feat_gallery = []
        sg_gallery = []
        print("开始分批提取特征...")
        for tiles_batch_cpu in tqdm.tqdm(torch.split(sat_tiles, split_size, dim=0), desc="Extracting Features"):
            tiles_batch_gpu = tiles_batch_cpu.to(self.device)
            with torch.no_grad():
                if with_sg:
                    output, output_sgs = model(tiles_batch_gpu, ret_sg=True)
                    sg_gallery.append(output_sgs)
                else:
                    output = model(tiles_batch_gpu, ret_sg=False)
            output = output[1] if isinstance(output, list) else output
            # 将计算结果立即移回CPU，以释放GPU显存
            feat_g = output.detach().cpu()
            feat_g = feat_g.view(feat_g.shape[0], -1) if len(feat_g.shape) > 2 else feat_g
            feat_gallery.append(feat_g)
        # --- 3. 在CPU上拼接所有批次的特征 ---
        feat_gallery = torch.cat(feat_gallery, dim=0)
        print("所有瓦片的特征已提取完成。")

        # get feat_querys from uavimgs
        feat_query,rc_query = [],[]
        sg_query = []
        for data in tqdm.tqdm(dataloader):
            uavimg_q, uav_rc = data[0].to(self.device),data[1]
            if with_sg:
                output,output_sgs = model(uavimg_q,ret_sg=True)
                sg_gallery.append(output_sgs)
            else:
                output = model(uavimg_q, ret_sg=False)
            feat_q = output[1] if type(output)==list else output
            feat_q =  feat_q.view(feat_q.shape[0],-1) if len(feat_q.shape)>2 else feat_q
            feat_query.append(feat_q.detach().cpu())
            rc_query.append(uav_rc)
            sg_query.append(output_sgs) if with_sg else None
        feat_query = torch.cat(feat_query,dim=0)
        rc_query = np.concatenate(rc_query,axis=0)

        # get georc_info
        georc_gallary = dataloader.dataset.sat_dataset.transform_nrc_to_georc(rc_gallery.reshape(-1,2)) #todo:
        sg_gallery = torch.cat(sg_gallery, dim=0).detach().cpu() if with_sg else None
        georc_query = dataloader.dataset.sat_dataset.transform_nrc_to_georc(rc_query)
        sg_query = torch.cat(sg_query, dim=0).detach().cpu() if with_sg else None

        # get gt_label fro perd
        from eval_recall_fm_salad import qurey_label_fm_gallery_nrc
        distrc_sats2uav, uav_labels_query = qurey_label_fm_gallery_nrc(rc_gallery.reshape(-1, 2), rc_query, dataloader.dataset.sat_dataset.halfimg_radius_nrc)
        rotdeg_fm_north_anticlock = dataloader.uav_dataset.rotdeg_fm_north_anticlock if hasattr(dataloader.dataset.uav_dataset,'rotdeg_fm_north_anticlock') else None

        # log the recall
        from eval_recall_fm_salad import compute_recall_from_feat
        d = compute_recall_from_feat(feat_query.contiguous(), feat_gallery.contiguous(), uav_labels_query,[1, 5, 20, 50, 200], faiss_gpu=False)
        info = "Recall"
        for k, v in d.items():
            info = info + f" @{k}:{v * 100:.2f} "
        self.logger.info(info) if hasattr(self, "logger") else None
        # write the recall to a txt
        p_txt2write = os.path.join(os.path.dirname(self.opt.load2test), f"recall_overlap{overlap}.txt")
        info2wirte = 'p_json=' + self.opt.p_satinfo_json
        info2wirte += f'\nn_refs={feat_gallery.shape[0]}'
        info2wirte += f'\nsatimgsize2crop={self.opt.satimgsize2crop}'
        info2wirte += f'\nrecall_m={dataloader.dataset.sat_dataset.halfimg_radius_meter}'
        # info2wirte += f'\nsample_overlap={overlap}'
        info2wirte += info
        with open(p_txt2write, 'a', encoding='utf-8') as f:
            f.write(info2wirte)

        if save_res:
            # save the result matrix:
            result = {'gallery_feat': feat_gallery.detach().cpu().numpy(),
                      'gallery_rc': rc_gallery,
                      'gallery_georc': georc_gallary,
                      'gallery_hw':np.array(rc_gallery.shape[:2]),
                      'query_feat': feat_query.detach().cpu().numpy(),
                      'query_rc': rc_query,
                      'query_latlon': georc_query,
                      'query_label': uav_labels_query,
                      'query_dist': distrc_sats2uav,
                      'radius_rc': dataloader.dataset.sat_dataset.halfimg_radius_nrc,
                      'radius_meter': dataloader.dataset.sat_dataset.halfimg_radius_meter,
                      }
            if with_sg:
                result['gallery_sg'] = sg_gallery
                result['query_sg'] = sg_query
                result['rotdeg_fm_north_anticlock'] = rotdeg_fm_north_anticlock

            suffix = f"_overlap{overlap}_radius{dataloader.dataset.sat_dataset.halfimg_radius_meter:.0f}m.mat"
            scipy.io.savemat( f"{self.opt.exp_dir}/{self.opt.exp_name}/{os.path.basename(self.opt.load2test).replace('.pth', suffix)}" ,result)


#########################与radom变换相关测试函数#####################
    def test_radon_wo_translate(self):
        model, dataloader = self._test_ready()

        # get feat_querys from uavimgs
        rc_query = []
        sg_query = []
        for data in tqdm.tqdm(dataloader):
            uavimg_q, uav_rc = data[0], data[1]
            _, output_sgs = model(uavimg_q.to(self.device), ret_sg=True)
            rc_query.append(uav_rc)
            sg_query.append(output_sgs)
        rc_query = np.concatenate(rc_query, axis=0)
        sg_query = torch.cat(sg_query, dim=0).detach().cpu()

        satimgs_positvie = torch.stack([dataloader.dataset.clip_satimg_fm_rc(rc) for rc in rc_query]).to(self.device)
        sg_positive = []
        for batch in torch.split(satimgs_positvie,  split_size_or_sections=32, dim=0):
            _, output_sg = model(batch, ret_sg=True)
            sg_positive.append(output_sg.detach().cpu())
        sg_positive = torch.cat(sg_positive, dim=0)

        from util_circorr_fm_radon import circorr_fm_radon,norm_rot
        # todo:decide the direction
        circor = circorr_fm_radon(sg_query.unsqueeze(1),sg_positive.unsqueeze(1)).numpy()
        # circor = circorr_fm_radon(sg_query.unsqueeze(1),sg_positive.unsqueeze(1)).numpy()
        pred_rot = np.argmax(circor,axis=-1)*5

        rotdeg_fm_north_anticlock = dataloader.dataset.rotdeg_fm_north_anticlock
        relrot_normed = norm_rot(rotdeg_fm_north_anticlock)[-pred_rot.shape[0]:]

        deg_tau = 15.
        deg_diff = np.abs(pred_rot - relrot_normed)
        recall_rot = ((deg_diff <= deg_tau).sum() + (deg_diff >= (360 - deg_tau)).sum()) / pred_rot.shape[0]
        info = f"RecallRot@{deg_tau:.1f}={recall_rot*100:.2f}\n"
        print(info)


    def test_radon_wo_translate_crossdomain(self):
        model, dataloader = self._test_ready()
        from util_circorr_fm_radon import norm_rot
        rotdeg_fm_north_anticlock = dataloader.dataset.rotdeg_fm_north_anticlock
        relrot_normed = norm_rot(rotdeg_fm_north_anticlock)[-len(dataloader.dataset.uavimg_paths_test):]
        import torchvision.transforms as transforms

        # get sg from uavimgs
        rc_query = []
        sg_query = []
        uavimgs_q = []
        for i,data in tqdm.tqdm(enumerate(dataloader)):
            uavimg_q, uav_rc = data[0], data[1]
            _, output_sgs = model(uavimg_q.to(self.device), ret_sg=True)
            uavimgs_q.append(uavimg_q)
            rc_query.append(uav_rc)
            sg_query.append(output_sgs)
        rc_query = np.concatenate(rc_query, axis=0)
        sg_query = torch.cat(sg_query, dim=0)

        # get sg roted to north from uavimgs
        uavimgs_q = torch.cat(uavimgs_q,dim=0)
        uavimgs_q_roted = [ transforms.functional.rotate(uavimg,-relrot_normed[i]) for i,uavimg in enumerate(uavimgs_q)]
        uavimgs_q_roted = torch.stack(uavimgs_q_roted, dim=0)
        sg_query_rot2north = []
        for data in torch.split(uavimgs_q_roted, split_size_or_sections=32, dim=0):
            _, output_sgs = model(data.to(self.device), ret_sg=True)
            sg_query_rot2north.append(output_sgs)
        sg_query_rot2north = torch.cat(sg_query_rot2north,dim=0)

        from util_circorr_fm_radon import circorr_fm_radon
        circor = circorr_fm_radon(sg_query.unsqueeze(1),sg_query_rot2north.unsqueeze(1)).cpu().numpy()
        pred_rot = np.argmax(circor,axis=-1)*5

        from vis_featmap import vis_rot_func
        uavimg_name = os.path.basename(dataloader.dataset.uavimg_paths_test[23])
        vis_rot_func(circor[23],rot_list=[i*5 for i in range(72)],p2save='visual_exp_imgs/'+f'circor_{uavimg_name[:-4]}.jpg')

        diff = (pred_rot - relrot_normed + 180) % 360 - 180 #Adjust difference to be in [-180, 180)
        angular_error = np.abs(diff)
        deg_tau = 15
        recall_rot = np.sum(angular_error <= deg_tau)/pred_rot.shape[0]
        info = f"RecallRot@{deg_tau:.1f}={recall_rot*100:.2f}\n"
        print(info)


#########################实验性函数，主要用于debug,需要手工挑选测试图像id#####################
    def test_xy(self,imgdir2save=None):
        model, dataloader=self._test_ready()

        # 选出测试图像
        query_id = 8735
        uav_rc = dataloader.dataset.uav_rcs[query_id]
        p2uav = dataloader.dataset.uavimg_paths[query_id]
        # query_id = 0
        # uav_rc = dataloader.dataset.uav_rcs_test[query_id]
        # p2uav = dataloader.dataset.uavimg_paths_test[query_id]
        uavimg_q = Image.open(p2uav)
        # satimg_q = dataloader.dataset.clip_satimg_fm_rc(uav_rc)

        # 得到测试图像对应所有旋转角度的特征
        angle_step = 10  # 每次旋转的角度
        num_rotations = 36  # 总共旋转的次数
        rot_list = [angle_step * i for i in np.arange(start=1,stop=num_rotations)]
        img_q_rots = [uavimg_q]
        img_q_rots += [uavimg_q.rotate(rot_angle, resample=Image.BICUBIC, expand=False) for rot_angle in rot_list]
        img_q_rots_t = torch.stack([dataloader.dataset.uav_transforms_test(img) for img in img_q_rots])

        output = model(img_q_rots_t.to(self.device))
        feat_q_rots = output[1] if type(output) == list else output
        feat_q_rots = feat_q_rots.view(feat_q_rots.shape[0], -1) if len(feat_q_rots.shape) > 2 else feat_q_rots
        feat_q_rots = F.normalize(feat_q_rots,dim=-1).detach().cpu()

        # get feat_gallary
        overlap = 0.75
        sat_tiles, rc_gallery = dataloader.dataset.clip_sat_unifrom(overlap=overlap)
        sat_tiles = sat_tiles.reshape(-1, * sat_tiles.shape[2:])
        split_size = 32
        feat_gallery = []
        for tiles in torch.split(sat_tiles, split_size, dim=0):
            output =  model(tiles)
            output = output[1] if type(output)==list else output
            feat_g =  output.detach().cpu()
            feat_g =  feat_g.view(feat_g.shape[0],-1) if len(feat_g.shape)>2 else feat_g
            feat_gallery.append(feat_g)
        feat_gallery = torch.cat(feat_gallery, dim=0) #.reshape(*rcs_girdcoord_center.shape[:2],-1)  # 形状 [N1*N2, D]
        feat_gallery = F.normalize(feat_gallery,dim=-1)

        # 得到测试图像对应所有旋转角度的在地图中的响应分布
        xydistb_rots = feat_gallery @ feat_q_rots.T
        ## 测试uavimg无旋转时
        # xydistb_0deg = xydistb_rots[:,0]
        # xydistb_0deg_max, xydistb_0deg_maxid = torch.max(xydistb_0deg, dim=-1)
        # predrc = rc_gallery.reshape(-1, 2)[xydistb_0deg_maxid]
        ## 测试uavimg有旋转时
        xydistb_argmaxdeg, ids = torch.max(xydistb_rots, dim=-1)
        xydistb_rots_max, xydistb_rots_maxid = torch.max(xydistb_argmaxdeg, dim=-1)
        predrc = rc_gallery.reshape(-1,2)[xydistb_rots_maxid]
        predrot = torch.argmax(xydistb_rots[xydistb_rots_maxid])*angle_step
        ## 计算误差
        predrc_denorm = predrc * max(rc_gallery.shape)
        gtrc_q_denorm = uav_rc * max(rc_gallery.shape)
        rcdist = np.linalg.norm(predrc_denorm - gtrc_q_denorm)

        # ---可视化---
        info_string = f'gtrc=({gtrc_q_denorm[0]:.1f},{gtrc_q_denorm[1]:.1f});predrc=({predrc_denorm[0]:.1f},{predrc_denorm[1]:.1f}); dist={rcdist:.2f} \n preduavrot={predrot}'
        fig, ax = plt.subplots()
        image_display = ax.imshow(xydistb_argmaxdeg.reshape(*rc_gallery.shape[:2]).detach().cpu().numpy(), cmap='coolwarm')
        fig.colorbar(image_display, ax=ax)  # 使用 'fig' 对象添加颜色条，并关联到 'ax'
        ax.scatter(gtrc_q_denorm[1], gtrc_q_denorm[0],  # 点的坐标
                    c='green',  # 颜色
                    s=5,  # 大小
                    marker='+',  # 形状
                    # label='gtrc',  # 标签 (用于图例)
                    zorder=10)  # 设置绘图顺序，确保点在最上层 (可选)
        ax.scatter(predrc_denorm[1], predrc_denorm[0],  # 点的坐标
                    c='blue',  # 颜色
                    s=5,  # 大小
                    marker='+',  # 形状
                    zorder=10)  # 设置绘图顺序，确保点在最上层 (可选)
        ax.text(0.02, 0.98, info_string, transform=ax.transAxes,  # <--- 在这里填入信息
                                  ha='left', va='top', fontsize=10, color='white',
                                  bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))
        ax.set_title('max(feat_gallery @ feat_query.T)')  # <--- 修改为你想要的标题

        # ax.legend()  #使用 'ax' 对象添加图例
        uavname = p2uav.split('/')[-1].split('.')[0]
        plt.savefig(f'/home/data/zwk/pyproj_DUAV_salad/xydistb_argmaxrot_{uavname}.jpg')
        plt.close(fig)
        # debug
        # torch.argmax(distb, dim=-1)
        # torch.argmax(distb_allrots[torch.argmax(distb, dim=-1)])
        # distb_allrots[torch.argmax(distb, dim=-1)]

    # def _mk_roted_imgs(self,uavimg_q,angle_step = 10, num_rotations = 36):
    #     # angle_step = 10  # 每次旋转的角度
    #     # num_rotations = 36  # 总共旋转的次数
    #     rot_list = [angle_step * i for i in np.arange(start=1,stop=num_rotations)]
    #     img_q_rots = [uavimg_q]
    #     img_q_rots += [uavimg_q.rotate(rot_angle, resample=Image.BICUBIC, expand=False) for rot_angle in rot_list]
    #     img_q_rots_t = torch.stack([self.dataloader.dataset.uav_transforms_test(img) for img in img_q_rots])
    #     return img_q_rots_t

    def test_rot(self,imgdir2save=None):
        model, dataloader = self._test_ready()

        query_id = 4546+4354
        uav_rc = dataloader.dataset.uav_rcs[query_id]
        p2uav = dataloader.dataset.uavimg_paths[query_id]
        # uav_rc = dataloader.dataset.uav_rcs_test[query_id]
        # p2uav = dataloader.dataset.uavimg_paths_test[query_id]
        uavimg_q = Image.open(p2uav)
        satimg_q = dataloader.dataset.clip_satimg_fm_rc(uav_rc)

        delta_deg = 10
        rot_list = [delta_deg * i for i in range(int(360 / delta_deg))]
        img_q_rots = [uavimg_q.rotate(rot_angle, resample=Image.BICUBIC, expand=False) for rot_angle in rot_list]
        # angle_step = 10  # 每次旋转的角度
        # num_rotations = 36  # 总共旋转的次数
        # rot_list = [angle_step * i for i in np.arange(start=0,stop=num_rotations)]
        # img_q_rots = [uavimg_q.rotate(rot_angle, resample=Image.BICUBIC, expand=False) for rot_angle in rot_list]
        img_q_rots_t = torch.stack([dataloader.dataset.uav_transforms_test(img) for img in img_q_rots])

        if len(img_q_rots_t)>36:
            split_size = 36
            feat_q_rots = []
            for tiles in torch.split(img_q_rots_t, split_size, dim=0):
                output_q =  model(tiles.to(self.device))
                output_q = output_q[1] if type(output_q)==list else output_q
                feat_q =  output_q.view(output_q.shape[0],-1) if len(output_q.shape)>2 else output_q
                feat_q_rots.append(feat_q)
            feat_q_rots = torch.cat(feat_q_rots, dim=0) #.reshape(*rcs_girdcoord_center.shape[:2],-1)  # 形状 [N1*N2, D]
        else:
            output_q = model(img_q_rots_t.to(self.device))
            feat_q_rots = output_q[1] if type(output_q) == list else output_q
            feat_q_rots = feat_q_rots.view(feat_q_rots.shape[0], -1) if len(feat_q_rots.shape) > 2 else feat_q_rots

        output_sat = model(satimg_q[None, ...].to(self.device))
        feat_s = output_sat[1] if type(output_sat) == list else output_sat
        feat_s = feat_s.view(feat_s.shape[0], -1) if len(feat_s.shape) > 2 else feat_s
        import torch.nn.functional as F
        rotdistb = F.normalize(feat_q_rots, dim=-1) @ F.normalize(feat_s, dim=-1).T

        rot_vals = rotdistb.squeeze().detach().cpu().numpy()
        angles_rad = np.radians(np.array(rot_list))  # Matplotlib 极坐标通常使用弧度
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})  # 关键：指定极坐标投影
        ax.plot(angles_rad, rot_vals, label='f(angle)')  # 绘制角度(弧度) vs 函数值(半径)
        ax.set_title(f"sim(feat(rot(uavimg,deg)),feat(satimg),predrot={np.argmax(rot_vals):.1f}deg", va='bottom')
        # ax.set_xlabel("Angle (radians)")  # 技术上是角度
        # ax.set_ylabel("Function Value (Radius)", labelpad=20)  # 技术上是半径
        # ax.set_rticks(np.arange(0, np.ceil(np.max(rot_vals)) + 0.5, 0.5))  # 设置半径刻度
        ax.set_theta_zero_location("N")  # 设置0度方向为北 (上方)
        ax.set_theta_direction(-1)  # 设置角度为顺时针方向 (可选)
        ax.grid(True)  # 显示网格
        # ax.legend() # 如果需要x/ylabel
        # plt.show()
        uavname = p2uav.split('/')[-1].split('.')[0]
        plt.savefig(f'/home/data/zwk/pyproj_DUAV_salad/rotdistb_{uavname}_10deg.jpg')
        plt.clf()

    def test_radon(self):
        model, dataloader = self._test_ready()
        # 选出测试图像
        query_id = 8819
        uav_rc = dataloader.dataset.uav_rcs[query_id]
        p2uav = dataloader.dataset.uavimg_paths[query_id]
        # uav_rc = dataloader.dataset.uav_rcs_test[query_id]
        # p2uav = dataloader.dataset.uavimg_paths_test[query_id]
        uavimg_q = Image.open(p2uav)
        uavimg_q = dataloader.dataset.uav_transforms_test(uavimg_q)
        satimg_q = dataloader.dataset.clip_satimg_fm_rc(uav_rc)
        input = torch.stack([uavimg_q,satimg_q])
        output = model(input.cuda())
        feat = output[1] if type(output) == list else output
        feat = feat.view(feat.shape[0], -1) if len(feat.shape) > 2 else feat


if __name__ == '__main__':
    torch.manual_seed(666)
    np.random.seed(2025)
    trainer = Trainer()
    trainer.train()
    # trainer.val()
    trainer.test(overlap=0.5)
    # trainer.test_xy()
    # trainer.output_test_res()
    # trainer.test_rot()
    # trainer.test_radon_wo_translate()
    # trainer.test_radon_wo_translate_crossdomain()