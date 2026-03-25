# -*- coding: utf-8 -*-
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from __future__ import print_function, division
import argparse
import torch
import tqdm
# from fontTools.ttLib.tables.otData import otData

# from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
# import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import time

from tool.util_mk_optimizer import make_optimizer
from train_img_encoder.nets_taskflow import make_model
from tool.utils_fm_duav import copyfiles2checkpoints, get_logger, set_seed
from tool.utils_fm_duav import load_network_wstate, save_network_wstate
import warnings
from losses.loss_cl import Loss
warnings.filterwarnings("ignore")

#import added:
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# var to selct:
# from datasets.make_dataloader import make_dataloader
# from datasets_custom.make_dataloder_classify import make_dataloader_train
from datasets_custom.make_dataloader_dsalad_nrc_center import make_dataloader
# from PIL import Image

import json
def json_dict(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON format for dictionary.")

def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    #about hardware
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--num_worker', default = 8, type=int, help='')
    parser.add_argument('--batchsize', default= 8, type=int, help='batchsize')
    parser.add_argument('--autocast', action='store_true', default=True, help='use mix precision')
    #about exp setting
    parser.add_argument('--p_satinfo_json',
                        default='/Localize/hsj/zwk/dataset_meta_XianganXmu/sat_imginfo.json',
                        type=str, help='training dir path')
    parser.add_argument('--p_uavinfo_json',
                        default='/Localize/hsj/zwk/dataset_meta_XianganXmu/uav_imginfo.json',
                        type=str, help='training dir path')
    parser.add_argument('--checkpoint', default="/home/data/zwk/pyproj_DUAV_salad/exps/exp27/epoch005.pth", type=str, help='path for testing')
    parser.add_argument('--load_from', default="", type=str, help='exps path for pre-loading')
    parser.add_argument('--exp_name', default='debug',
                        type=str, help='the experiment name that will be saved in exps dir in the root')
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--val', action='store_true', default = False)
    parser.add_argument('--val_freq', default = 10, type=int)
    parser.add_argument('--tensorboard', action='store_true', default = True)
    parser.add_argument('--n_satrand_per_uav', default=8, type=int, help='will be used in dataset')
    #about data argument
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=224, type=int, help='height')
    parser.add_argument('--w', default=224, type=int, help='width')
    parser.add_argument('--rr', default="uav", type=str, help='random rotate')
    parser.add_argument('--ra', default="satellite", type=str, help='random affine')
    parser.add_argument('--re', default="satellite", type=str, help='random erasing')
    parser.add_argument('--cj', default="no", type=str, help='color jitter')
    parser.add_argument('--erasing_p', default=0.3, type=float,
                        help='random erasing probability, in [0,1]')
    parser.add_argument('--DA', action='store_true',
                        help='use Color Data Augmentation')
    #about learning setting
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
    parser.add_argument('--num_epochs', default=50, type=int, help='total epoches for training')
    parser.add_argument('--warm_epoch', default=0, type=int,
                        help='the first K epoch that needs warm up')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    #about networks
    parser.add_argument('--w_classify', default=False, action='store_true', help='')
    parser.add_argument('--cls_loss', default="CELoss", type=str, help='loss type of representation learning')
    parser.add_argument('--kl_loss', default="KLLoss", type=str, help='loss type of mutual learning')
    parser.add_argument('--feature_loss', nargs='+', default=["WeightedSoftTripletLoss"],
                        help='"InfoNceLoss","TripletLoss","HardMiningTripletLoss","SameDomainTripletLoss","WeightedSoftTripletLoss","ContrastiveLoss"')
    parser.add_argument('--backbone', default="ViTB-224", type=str, help='ViTB-224;dinov2_vitb14;convnext')
    parser.add_argument('--head', default="salad", type=str, help='salad;FSRA;LPN')
    parser.add_argument('--block', default=2, type=int, help='') #will by used when headF=FSRA,LPN,NetVLAD,NeXtVLAD
    parser.add_argument('--num_bottleneck', default=512, type=int, help='the dimensions for embedding the feature')
    parser.add_argument('--head_pool', default="avg", type=str, help='head pooling type for applying') #will by used when head=SingleBranch
    parser.add_argument('--wcls_token', default=False, type=bool) #will by used when head=SingleBranch
    parser.add_argument('--norm_output', default=True, type=bool)

    opt = parser.parse_args()
    group_info = {
        'Hardware Settings': ['gpu_ids', 'num_worker', 'batchsize', 'autocast'],
        'Experiment Settings': ['exp_name', 'p_satinfo_json','p_uavinfo_json','load_from', 'checkpoint', 'val', 'val_freq', 'save_freq', 'tensorboard'],
        'Data Augmentation Settings': ['pad', 'h', 'w', 'rr', 'ra', 're', 'cj', 'erasing_p', 'DA'],
        'Learning Settings': ['warm_epoch', 'num_epochs', 'droprate','optimizer','lr_sched'],
        'Network Settings': ['w_classify','block', 'cls_loss', 'feature_loss', 'kl_loss', 'num_bottleneck', 'backbone', 'head', 'head_pool','norm_output']
    }
    opt.group_dict = group_info
    print(opt)
    return opt

import yaml
import os
import scipy
class Trainer(object):
    def __init__(self):
        self.opt = get_parse()

        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.opt.use_gpu = True
        else:
            device = torch.device("cpu")
            self.opt.use_gpu = False
        self.deivce = device

    def train(self):
        opt = self.opt
        self.model = make_model(self.opt)
        if self.opt.use_gpu:
            self.model = self.model.cuda()
        model = self.model
        self.optimizer_ft, self.exp_lr_scheduler = make_optimizer(self.model, self.opt)
        optimizer,scheduler = self.optimizer_ft,self.exp_lr_scheduler

        # save_network_wstate(opt.exp_name,model,optimizer,scheduler,0)
        if len(opt.load_from)>0:
            begin_epoch = load_network_wstate(opt.load_from, model, optimizer, scheduler)
        else:
            begin_epoch = 0

        # backup files
        if not (len(opt.load_from)>0 and os.path.dirname(opt.load_from).split('/')[-1]==opt.exp_name):
           copyfiles2checkpoints( self.opt )
           # config logger
           logger = get_logger(
               "exps/{}/train.log".format(opt.exp_name))
           # config tensorborad
           writer = SummaryWriter("exps/{}/train_tensorboard.log".format(opt.exp_name)) if opt.tensorboard else None

        # config dataloader
        self.dataloader = make_dataloader(self.opt,stage='train')
        dataloader = self.dataloader

        # ready to trian
        num_epochs = opt.num_epochs
        since = time.time()
        scaler = GradScaler()
        nnloss = Loss(opt)
        step = 0
        for epoch in range(begin_epoch,num_epochs):
            logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            logger.info('-' * 50)

            model.train(True)  # Set model to training mode

            for it,data in tqdm.tqdm(enumerate(dataloader)):
                # time_it10 = time.time()
                # 获取输入无人机和卫星数据
                imgs_d, imgs_s, imgs_s_rand, rcs_d, rcs_rand = data
                imgs = torch.cat([imgs_d,imgs_s,imgs_s_rand],dim=0).to(self.deivce)
                # rcs = torch.concatenate([rcs_d,rcs_d,rcs_rand],dim=0).to(self.deivce)
                n_d = imgs_d.shape[0]

                # 梯度清零
                optimizer.zero_grad()
                with autocast():
                    if opt.wcls_token:
                        outputs, clses = model(imgs)
                    else:
                        outputs = model(imgs)
                outputs = torch.cat(outputs,dim=-1) if type(outputs) == list else outputs

                outputs = F.normalize(outputs,dim=-1) if opt.norm_output else outputs
                loss_dict = nnloss.forward(outputs[:n_d],outputs[n_d:2*n_d],outputs[2*n_d:])

                # todo:训练时是f_uav@f_sat 还是 [f_uav,f_sat]@[f_uav,f_sat]更有效？
                # todo:仅仅使用clses测评未经训练的Dinov2的检索性能？
                # clses = F.normalize(clses, dim=-1)
                # temperature = 0.1
                # # feat_mat = outputs[0:1] @ outputs[1:].T / temperature
                # feat_mat = clses[0:1] @ clses[1:].T / temperature
                # feat_mat_np = feat_mat.detach().cpu().numpy()
                # info_nce_loss = -torch.log(
                #     torch.exp(feat_mat[0, 0]) / torch.sum(torch.exp(feat_mat[0, 1:]), dim=-1)).mean()

                # time_netbackwad_begin = time.time()
                # 反向传播
                if opt.autocast:
                    scaler.scale(loss_dict['all']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict['all'].backward()
                    optimizer.step()
                # print("backward_time:{}".format(time.time()-start_time))
                # time_netbackwad_end =  time.time() - time_netbackwad_begin
                # print(f"time_nerforwad={time_netbackwad_end:.6f}")

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

            str2log = ''
            for k, v in loss_dict.items():
                str2log += f'{k}={v.item():.3f}; '
            str2log += f'epcoh={epoch}'
            logger.info(str2log)
            # if opt.tensorboard:
            #     writer.add_scalars('losses', loss_dict, epoch)
            time_elapsed = time.time() - since
            since = time.time()
            logger.info('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 6))

    def val(self):
        pass

    def test(self):
        opt = self.opt
        # load the training config to update opt
        config_path = os.path.dirname(opt.checkpoint) + os.sep + 'opts.yaml' #todo: add opt.checkpoint
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        for group_dict_key, group_dict in config.items():
            if group_dict_key=='Network Settings':
                for cfg, value in group_dict.items():
                        setattr(opt, cfg, value)
            else:
                for cfg, value in group_dict.items():
                    if not hasattr(opt, cfg):
                        setattr(opt, cfg, value)

        self.model = make_model(self.opt)
        if self.opt.use_gpu:
            self.model = self.model.cuda()
        model = self.model
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint["model_state"]) if "model_state" in checkpoint else model.load_state_dict(checkpoint)
        model = model.eval()

        dataloader = make_dataloader(opt,stage='test') if not hasattr(self,'dataloader') else self.dataloader
        overlap = 0.75
        sat_tiles, rc_gallery = dataloader.dataset.split_sat_unifrom(overlap=overlap)
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
        latlon_gallary = dataloader.dataset.rowcol_to_latlon(rc_gallery)

        feat_query,rc_query = [],[]
        for data in tqdm.tqdm(dataloader):
            uavimg_q, uav_rc = data[0],data[1]
            output = model(uavimg_q.cuda())
            feat_q = output[1] if type(output)==list else output
            feat_q =  feat_q.view(feat_q.shape[0],-1) if len(feat_q.shape)>2 else feat_q
            feat_query.append(feat_q.detach().cpu())
            rc_query.append(uav_rc)
        feat_query = torch.concatenate(feat_query,dim=0)
        rc_query = np.concatenate(rc_query,axis=0)
        latlon_query = dataloader.dataset.rowcol_to_latlon(rc_query)

        from datasets_custom.make_dataloader_dsalad_nrc_center import qurey_label_fm_gallery_rc
        distrc_sats2uav,uav_labels_query=qurey_label_fm_gallery_rc(rc_gallery.reshape(-1,2),rc_query,dataloader.dataset.halfimg_radius_rc)

        result = {'gallery_feat': feat_gallery.numpy(),
                  'gallery_rc': rc_gallery.reshape(-1,2),
                  'gallery_latlon': latlon_gallary.reshape(-1,2),
                  'gallery_shape':np.array(feat_gallery.shape[:2]),
                  'query_feat': feat_query.numpy(),
                  'query_rc': rc_query,
                  'query_latlon': latlon_query,
                  'query_label': uav_labels_query,
                  'query_dist': distrc_sats2uav,
                  'radius_rc': dataloader.dataset.halfimg_radius_rc,
                  'radius_met': dataloader.dataset.halfimg_radius_met,
                  }
        suffix = f"_overlap{overlap}_radius{dataloader.dataset.halfimg_radius_met:.0f}m.mat"
        scipy.io.savemat(opt.checkpoint.replace('.pth', suffix), result) # todo: change to exp dir

if __name__ == '__main__':
    set_seed(666)

    trainer = Trainer()
    trainer.train()
    # trainer.test()
