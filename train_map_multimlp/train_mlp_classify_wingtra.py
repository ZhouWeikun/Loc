import torch
import argparse
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from mertic_learning import LpDistance
import numpy as np
import torchvision.transforms as T
import yaml

from get_img_encoder import ImgEncoder
# from dataset_map_learner_xmu import SatMapDataset, UAVDataset
# from dataset_map_learner_gta import SatMapDataset, UAVDataset
from dataset_map_learner_wingtra import SatDataset, UAVDataset
torch.manual_seed(2025)
np.random.seed(2025)

# from vis_featmap import vis_multi_featmap,vis_single_featmap
from eval_recall_fm_salad import compute_recall_by_label
from loc_utils import agg_seq_pdf,compute_agged_pred_4neighbors_id,find_4neighbors_topleft,compute_agged_pred_nneighbors_id,find_nneighbors_topleft

def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")
    parser.add_argument('--exp_name', default='debug_zuchwil', type=str)
    parser.add_argument('--exps_dir', default='/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps', type=str)
    parser.add_argument('--p_satinfo_json',
                        default='/home/data/zwk/data_uavimgs_wingtra/Zurich/SWISSIMAGE2022_blocks132_res05m.json',
                        type=str, help='training sat dir path')
    parser.add_argument('--p_uavinfo_json',
                        default='/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info.json',
                        type=str, help='training uav dir path')
    parser.add_argument('--p_img_encoder_cfg',
                        default='/home/data/zwk/pyproj_DUAV_salad_6.4/exps/zurich_132km_05res_best/opts_wingtra.yaml',
                        type=str, help='training uav dir path')
    parser.add_argument('--p_img_encoder_ckpt',
                        default='/home/data/zwk/pyproj_DUAV_salad_6.4/exps/zurich_132km_05res_best/epoch002.pth',
                        type=str, help='training uav dir path')
    parser.add_argument('--imgsize2net',default = 224, type=int, help='the img size to net')
    parser.add_argument('--satimgsize2crop',default = 244, type=int, help='the satimg cliped from tif') #according to the img_encoder config
    parser.add_argument('--grid_scale_on_satimgsize2crop', default=2.5, type=int, help='be used to determine the size of each classified area')
    parser.add_argument('--batchsize_sat', default=128, type=int, help='batchsize')
    parser.add_argument('--batchsize_uav', default=8, type=int, help='batchsize')
    parser.add_argument("--n_epoch", nargs="?", type=int, default=100, help="Number of training steps")
    parser.add_argument('--num_worker', default=8, type=int)
    parser.add_argument('--tensorboard', action='store_true', default = False)
    parser.add_argument('--len_neighbors', default = 2, type=int, help='4 or 9 or 25...') #be used during test
    parser.add_argument('--seq_agg_len', default = 5, type=int, help='2,3,4...') #be used during test
    parser.add_argument('--ckpt2test', default="/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_zuchwil_5/epoch_3.pth", type=str, help='path for testing') # for testing
    parser.add_argument('--ckpt2train', default="", type=str, help='exps path for pre-loading') #for continuing training

    return parser.parse_args()


class Trainer(object):
    def __init__(self,):
        # config the args:
        self.args = get_args()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            self.args.use_gpu = True
        else:
            device = torch.device("cpu")
            self.args.use_gpu = False
        self.device = device

        # config the trained img_encoder:
        self.img_encoder = ImgEncoder(self.args.p_img_encoder_cfg,self.args.p_img_encoder_ckpt)

        # config the datasets:
        self.sat_dataset = SatDataset(
            p_satinfo_json=self.args.p_satinfo_json,
            satimgsize2crop=self.args.satimgsize2crop,
            imgsize2net=224,
        )
        self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset, batch_size=self.args.batchsize_sat,
                                                          num_workers=self.args.num_worker,
                                                          pin_memory=True, shuffle=True, drop_last=True,
                                                          persistent_workers=True)
        self.uav_dataset_train = UAVDataset(
            p_uavinfo_json = self.args.p_uavinfo_json,
            imgsize2net = 224,
            stage='train',
            # trans_latlon2nrc_func=self.sat_dataset.transfrom_latlon_to_nrc,
        )
        self.uav_dataset_test = UAVDataset(
            p_uavinfo_json=self.args.p_uavinfo_json,
            imgsize2net = 224,
            stage='test',
            # trans_latlon2nrc_func=self.sat_dataset.transfrom_latlon_to_nrc,
        )
        self.uav_dataset_train.uav_nrcs = self.sat_dataset.transfrom_georc_to_nrc(self.uav_dataset_train.uav_georcs,source_epsg_code=self.uav_dataset_train.epsg_code)
        self.uav_dataset_test.uav_nrcs = self.sat_dataset.transfrom_georc_to_nrc(self.uav_dataset_test.uav_georcs,source_epsg_code=self.uav_dataset_train.epsg_code)
        self.uav_dataset_train.split_uav_dataset()
        self.uav_dataset_test.split_uav_dataset()

        self.uav_dataloader_train = torch.utils.data.DataLoader(self.uav_dataset_train, batch_size=self.args.batchsize_uav,
                                                                num_workers=self.args.num_worker,
                                                                pin_memory=True, shuffle=True, drop_last=False,
                                                                persistent_workers=True)
        self.uav_dataloader_test = torch.utils.data.DataLoader(self.uav_dataset_test, batch_size=self.args.batchsize_uav,
                                                               num_workers=self.args.num_worker,
                                                               pin_memory=True, shuffle=False, drop_last=False,
                                                               persistent_workers=True)


    def cfg_mlps_w_dataset(self,update_args=False):
        if update_args:
            p2yaml = os.path.join(os.path.dirname(self.args.ckpt2test), "args.yaml")
            with open(p2yaml, 'r') as stream:
                config = yaml.load(stream, Loader=yaml.FullLoader)
            self.args.grid_scale_on_satimgsize2crop = config['grid_scale_on_satimgsize2crop']
            self.args.satimgsize2crop = config['satimgsize2crop']

        self.nrcs_grid_center = self.sat_dataset.mk_coord_grid(split_by='pix_per_grid_hw',
                                                               pix_per_grid_hw=(
                                                               self.args.grid_scale_on_satimgsize2crop * self.args.satimgsize2crop,
                                                               self.args.grid_scale_on_satimgsize2crop * self.args.satimgsize2crop),
                                                               random=False).reshape(-1, 2).to(self.device)
        self.nrcs_gird_boundary = self.sat_dataset.nrcs_grid_boundary
        self.n_grid_hw = (self.sat_dataset.n_grid_h, self.sat_dataset.n_grid_w)
        self.dist2gaussian = torch.distributions.Normal(loc=0,
                                                        scale=self.sat_dataset.grid_cell_radius * 1.)  # scale=sigma, 2sigma that contains 99% pdf = grid_cell_radius
        self.eucdist_computer_feat = LpDistance(normalize_embeddings=True)
        self.eucdist_computer_rc = LpDistance(normalize_embeddings=False)

        # create the map_encoder:
        from get_pos_encoder import PositionalEncoder
        pos_encoder = PositionalEncoder(multires=0, input_dims=2)
        self.pos_encoder = pos_encoder
        from get_mlps_classify import create_mlp, init_weights, MultiMLP
        # input_dim = self.img_encoder.get_output_dim() + pos_encoder.out_dim
        input_dim = 768
        mlp_dropout_p = 0.
        # create the map_encoder with multi_mlps
        # map_encoder = MultiMLP(self.nrcs_grid_center.shape[0], input_dim=input_dim,
        #                             mlp_hidden_dims=[256, 128, 64],  # [ 128, 32, 8]
        #                             mlp_activation_fn=nn.LeakyReLU, mlp_norm_type='layer',
        #                             mlp_dropout_p=mlp_dropout_p,
        #                             mlp_init_method='kaiming', mlp_init_nonlinearity='leaky_relu',
        #                             device=self.device)
        # create the map_encoder with single mlp
        dims = [input_dim, 512, 512, self.nrcs_grid_center.shape[0]]
        activation = nn.LeakyReLU
        map_encoder = create_mlp(dims, activation_fn=activation, norm_type='layer').to(self.device)
        map_encoder.apply(lambda m: init_weights(m, method='kaiming', nonlinearity='leaky_relu'))

        self.map_encoder = map_encoder
        self.map_encoder.mlp_dropout_p = mlp_dropout_p

        #wrting args for saving exp_info
        self.args.n_grid_h = self.n_grid_hw[0]
        self.args.n_grid_w = self.n_grid_hw[1]
        self.args.satimgsize_per_aera = self.sat_dataset.satimgsize2crop*self.args.grid_scale_on_satimgsize2crop

        # debug,vis the gird
        # from vis_loc_res import vis_rcs_on_tif,draw_nrc_grid_on_image
        # pts = torch.cat([self.nrcs_grid_center.detach().cpu(), self.sat_dataset.nrcs_grid_boundary.reshape(-1, 2)]).numpy()
        # dir2vis = '/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/vis_grid'
        # vis_rcs_on_tif(np.array(self.sat_dataset.satmap), self.nrcs_grid_center.detach().cpu().numpy(),
        #                p2save=dir2vis+'/grid_centers.png',
        #                color='red')
        # draw_nrc_grid_on_image(np.array(self.sat_dataset.satmap),
        #                    self.sat_dataset.nrcs_grid_boundary.detach().cpu().numpy(),
        #                     p2save = dir2vis+'/grid_boundary_zurich_132km2.png',
        #                    line_width=0.5,
        #                    )
        # vis_rcs_on_tif(np.array(self.sat_dataset.satmap), pts,
        #                p2save=dir2vis+'/grid_cneter&boundary_zhchwil_30mk2',
        #                color='blue')


    def train(self):
        # config the mlps according to the grids with the dataset
        self.cfg_mlps_w_dataset()

        # config the vals
        args = self.args
        device = self.device
        map_encoder = self.map_encoder
        img_encoder = self.img_encoder
        eucdist_computer_rc = self.eucdist_computer_rc
        grid_centers = self.nrcs_grid_center
        dist2gaussian = self.dist2gaussian

        # config the optimizer
        optimizer_cfg = {
            "otype": "Adam",
            "lr": 1e-2,
            "beta1": 0.9,
            "beta2": 0.99,
            "eps": 1e-8,
            "l2_reg": 1e-4
        }
        optimizer = torch.optim.Adam(map_encoder.parameters(), lr=optimizer_cfg['lr'],
                                     betas=(optimizer_cfg['beta1'], optimizer_cfg['beta2']),
                                     eps=optimizer_cfg['eps'], weight_decay=optimizer_cfg['l2_reg'])

        # load the ckpt and config if necessary
        epoch_begin = 0
        if args.ckpt2train != "":
            checkpoint = torch.load(args.ckpt2train, map_location=device)  # map_location确保能正确加载到CPU或GPU
            map_encoder.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_begin = checkpoint['epoch']

        # config for recording exp & backup files
        from tool.utils import get_unique_exp_dir,copyfiles2checkpoints_map_learner
        self.args.exp_name = get_unique_exp_dir(self.args.exps_dir,self.args.exp_name)
        expdir2save = "{}/{}".format(self.args.exps_dir,self.args.exp_name)
        if not os.path.exists(expdir2save):
            os.mkdir(expdir2save)
        self.writer = SummaryWriter(f"{expdir2save}/{args.exp_name}/train_tensorboard.log") if args.tensorboard else None
        from tool.utils import  get_logger
        self.logger = get_logger("{}/{}/train.log".format(self.args.exps_dir,self.args.exp_name),'trainer_logger')
        copyfiles2checkpoints_map_learner( self.args )

        # ready to train
        for epoch in torch.range(epoch_begin,args.n_epoch,dtype=torch.uint8):
            for it, data in tqdm(enumerate(self.sat_dataloader)):
                imgs,nrcs = data
                uav_imgs,uav_nrcs = next(iter(self.uav_dataloader_train))
                nrcs = torch.concatenate([nrcs,uav_nrcs],dim=0).to(device)
                imgs = torch.concatenate([imgs,uav_imgs],dim=0).to(device)

                feats = img_encoder.model(imgs)
                input = feats
                output = map_encoder(input)
                pred_pdf = torch.softmax(output,dim = -1)

                nrcs2centers = eucdist_computer_rc(nrcs,grid_centers)
                gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers))
                gt_pdf = gt_pdf / gt_pdf.sum(dim = -1, keepdim = True)
                loss = - gt_pdf * torch.log(pred_pdf)
                loss = loss.sum(dim=-1).mean()

                # debug
                # featmap = pred_pdf[:9].reshape(-1,8,10).detach().cpu().numpy()
                # vis_multi_featmap(featmap,p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/pred_pdf_train.png',interpolation='nearest')
                # featmap = gt_pdf[:9].reshape(-1,8,10).detach().cpu().numpy()
                # vis_multi_featmap(featmap,p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/gt_pdf_train.png',interpolation='nearest')

                optimizer.zero_grad()
                loss.backward()
                # --- 梯度检查点 ---
                # 在这里检查梯度
                # total_norm = 0
                # for p in map_encoder.parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.data.norm(2)  # 计算单个参数梯度的L2范数
                #         total_norm += param_norm.item() ** 2  # 累加平方
                # total_norm = total_norm ** (1. / 2)
                # 打印当前batch的梯度总范数
                # print(f"Loss: {loss.item():.4f} | Gradient Norm: {total_norm:.4f}")
                # ------------------
                optimizer.step()
                print(f"loss={loss:.4f}")
                self.writer.add_scalar('loss', loss, it) if self.writer is not None else None

            # test and log
            self.test(eval_on='uav', load_ckpt=False)
            self.map_encoder.train()
            self.img_encoder.model.train()
            # self.writer.add_scalar('recall1', recall1, epoch) if self.writer is not None else None
            # self.writer.add_scalar('recall1_rel_dist', recall1_rel_dist, epoch) if self.writer is not None else None
            self.logger.info(f"ep={epoch} finished ") if hasattr(self,'logger') else None

            # sava the ckpt
            # if epoch % 5 == 0:
            dir2save = f"{self.args.exps_dir}/{self.args.exp_name}"
            torch.save({
                'epoch': epoch,
                'model_state_dict': map_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,  # 可选：保存最后一个 batch 的损失值
            }, os.path.join(dir2save, f"epoch_{epoch}.pth"))


    def test(self, eval_on='uav', load_ckpt=False, fine_loc=False, save=False):
        # config the mlps according to the grids with the dataset
        self.cfg_mlps_w_dataset(update_args=True) if not hasattr(self, 'map_encoder') else None

        with torch.no_grad():
            # config the vals
            args = self.args
            device = self.device
            sat_dataloader = self.sat_dataloader
            uav_dataloader = self.uav_dataloader_test
            map_encoder = self.map_encoder
            img_encoder = self.img_encoder
            eucdist_computer_rc = self.eucdist_computer_rc
            grid_centers = self.nrcs_grid_center
            dist2gaussian = self.dist2gaussian
            eucdist_computer_feat = self.eucdist_computer_feat
            resize_transform = T.Resize(self.args.imgsize2net)

            # load the ckpt and config if necessary
            if load_ckpt:
                checkpoint = torch.load(args.ckpt2test, map_location=device)  # map_location确保能正确加载到CPU或GPU
                map_encoder.load_state_dict(checkpoint['model_state_dict'])
            img_encoder.model.eval()
            map_encoder.train() if map_encoder.mlp_dropout_p > 0 else map_encoder.eval()

            # select the dataloader for test
            if eval_on == 'uav':
                dataloader = uav_dataloader
            else:
                dataloader = sat_dataloader

            # getting the output for test
            q_label_list = []
            pred_pdf_list = []
            pred_nrc_list = []
            for it, data in tqdm(enumerate(dataloader)):
                imgs, nrcs = data
                imgs, nrcs = imgs.to(device), nrcs.to(device)

                feats_q = img_encoder.model(imgs)
                input = feats_q
                if self.map_encoder.mlp_dropout_p > 0:
                    batch_predictions = []
                    for _ in range(50):
                        output = map_encoder(input)
                        pred_pdf = torch.softmax(output, dim=-1)
                        batch_predictions.append(pred_pdf)
                    stacked_preds = torch.stack(batch_predictions, dim=1)
                    mean_prediction = torch.mean(stacked_preds, dim=1)
                    std_prediction = torch.std(stacked_preds, dim=1)
                    pred_pdf = std_prediction
                else:
                    output = map_encoder(input)
                    pred_pdf = torch.softmax(output, dim=-1)
                pred_pdf_list.append(pred_pdf)

                nrcs2centers = eucdist_computer_rc(nrcs, grid_centers)
                gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers))
                gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)
                q_labels = torch.argmax(gt_pdf, dim=-1)
                q_label_list.append(q_labels)

                # fine loc:
                if fine_loc:
                    len_neighbors = self.args.len_neighbors
                    id_neighbors_flat, id_neighbors_2d = compute_agged_pred_nneighbors_id(
                        pred_pdf.reshape(-1, self.n_grid_hw[0], self.n_grid_hw[1]), len_neighbors, ret_2d=True)
                    nrc_topleft_neighbors = self.sat_dataset.nrcs_grid_boundary[
                        id_neighbors_2d[:, 0, 0].cpu(), id_neighbors_2d[:, 0, 1].cpu()]
                    nrc_rightbottom_neighbors = nrc_topleft_neighbors + torch.stack(
                        [self.sat_dataset.gird_cell_h, self.sat_dataset.gird_cell_w])[None, ...] * len_neighbors

                    nrc_topleft_previous = None
                    for i, nrc_topleft in enumerate(nrc_topleft_neighbors):
                        # --- 1. 判断当前坐标是否需要处理 ---
                        # 条件：是第一个元素，或者当前元素与上一个已处理的元素不同
                        should_process = (i == 0) or (torch.abs(nrc_topleft - nrc_topleft_previous).sum() > 1e-5)

                        # --- 2. 如果需要处理，则执行特征提取 ---
                        if should_process:
                            # 这个代码块现在是唯一的，不再重复
                            sat_tiles_tensor, nrcs_grid_center = self.sat_dataset.sample_sats_in_rect(
                                nrc_topleft,  # 使用当前的 nrc_topleft
                                nrc_rightbottom_neighbors[i],
                                satimgsize2crop=self.args.satimgsize2crop,
                                type2clip='tensor'
                            )
                            sat_tiles_tensor = resize_transform(
                                sat_tiles_tensor.reshape(-1, *sat_tiles_tensor.shape[2:]))

                            split_size = 256
                            feat_gallery = []
                            print("开始为新区域分批提取特征...")
                            for tiles_batch_cpu in tqdm(torch.split(sat_tiles_tensor, split_size, dim=0),
                                                        desc="Extracting Features"):
                                tiles_batch_gpu = tiles_batch_cpu.to(self.device)
                                with torch.no_grad():  # 在推理时使用 no_grad() 以节省显存和加速
                                    output = self.img_encoder.model(tiles_batch_gpu)

                                feat_g = output.detach().cpu()
                                feat_g = feat_g.view(feat_g.shape[0], -1) if len(feat_g.shape) > 2 else feat_g
                                feat_gallery.append(feat_g)

                            feat_gallery = torch.cat(feat_gallery, dim=0)
                            print("特征提取完成。")
                            # 更新“上一个已处理”的坐标
                            nrc_topleft_previous = nrc_topleft

                        # 计算定位的概率分布
                        feat_dist = eucdist_computer_feat(feats_q[i][None, :].cpu(), feat_gallery)
                        dist2prob_scale = 2.0
                        nrc_distb = torch.exp(-dist2prob_scale * feat_dist).reshape(*nrcs_grid_center.shape[:2])
                        nrc_distb_normed = nrc_distb / nrc_distb.sum()
                        # 计算精细定位结果
                        pred_r, pred_c = torch.where(nrc_distb_normed == nrc_distb_normed.max())
                        pred_nrc_in_grid = torch.tensor(
                            [pred_r / nrc_distb.shape[0], pred_c / nrc_distb.shape[1]]) * torch.tensor(
                            [self.sat_dataset.gird_cell_h, self.sat_dataset.gird_cell_w]) * len_neighbors
                        pred_nrc = pred_nrc_in_grid + nrc_topleft_neighbors[i]
                        pred_nrc_list.append(pred_nrc)

                        # debug 验证定位结果, #todo: checkout the uav_nrc
                        # from vis_featmap import vis_single_featmap,add_pts2map
                        # dir2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_zuchwil_3/exp_vis'
                        # p_uavimg = self.uav_dataset_test.uavimg_paths_test[i]
                        # uavimg_name = p_uavimg.split('/')[-1].split('.')[0][-8:]
                        # p2save_xyprob = f'{dir2save}/xy_prob_{uavimg_name}.png'
                        # fig,ax = vis_single_featmap(nrc_distb_normed,p2save=p2save_xyprob,return_handles=True)
                        # gt_nrc_in_grid = nrcs[i].cpu()-nrc_topleft_neighbors[i]
                        # gt_rc = gt_nrc_in_grid /len_neighbors / torch.tensor([self.sat_dataset.gird_cell_h,self.sat_dataset.gird_cell_w]) * torch.tensor([nrc_distb_normed.shape[0],nrc_distb_normed.shape[1]])
                        # pred_rc = torch.tensor([pred_r,pred_c])
                        # add_pts2map(fig,ax,torch.stack([gt_rc,pred_rc]),labels=['gt','pred'],p2save=p2save_xyprob)
                        # from matplotlib import pyplot as plt
                        # from PIL import Image
                        # uavimg = Image.open(p_uavimg)
                        # plt.imshow(uavimg)
                        # plt.savefig(f'{dir2save}/uavimg_{uavimg_name}.png')
                        # plt.close()
                        # satimg_rect = self.sat_dataset.crop_rect_satimg(nrc_topleft=nrc_topleft_neighbors[i],nrc_rightbottom=nrc_rightbottom_neighbors[i], type='np')
                        # plt.imshow(satimg_rect)
                        # plt.savefig(f'{dir2save}/rect_satimg_{uavimg_name}.png')
                        # plt.close()
                        # uav_nrc =  self.uav_dataset_test.uav_nrcs_test[i]
                        # satimg = self.sat_dataset.crop_satimg_by_nrc(uav_nrc,type='np')
                        # plt.imshow(satimg)
                        # plt.savefig(f'{dir2save}/satimg_{uavimg_name}.png')
                        # plt.close()

            q_label_list = torch.cat(q_label_list).cpu().numpy()
            pred_pdf_list = torch.cat(pred_pdf_list)

            # compute the metrics according to the output:
            pred_id_sorted = torch.sort(pred_pdf_list, descending=True, dim=-1)[1]
            single_recall = compute_recall_by_label(q_label_list, pred_id_sorted.detach().cpu().numpy(),
                                                    [1, 2, 3, 4, 5], title='single_recall')

            len_neighbors = self.args.len_neighbors
            k_values = [i + 1 for i in np.arange(len_neighbors ** 2)]
            id_neighbors_1d, id_neighbors_2d = compute_agged_pred_nneighbors_id(
                pred_pdf_list.reshape(-1, self.n_grid_hw[0], self.n_grid_hw[1]), len_neighbors, ret_2d=True)
            neighbors_recall = compute_recall_by_label(q_label_list, id_neighbors_1d.cpu().numpy(), k_values,
                                                       title=f'{len_neighbors ** 2}neighbors_recall')

            seq_agg_len = self.args.seq_agg_len
            pred_pdf_agged = agg_seq_pdf(pred_pdf_list, window_len=seq_agg_len)
            id_neighbors_agged_1d, id_neighbors_agged_2d = compute_agged_pred_nneighbors_id(
                pred_pdf_agged.reshape(-1, self.n_grid_hw[0], self.n_grid_hw[1]), len_neighbors, ret_2d=True)
            seq_neighbors_recall = compute_recall_by_label(q_label_list[-id_neighbors_agged_1d.shape[0]:],
                                                           id_neighbors_agged_1d.detach().cpu().numpy(), k_values,
                                                           title=f'{seq_agg_len}agged_{len_neighbors ** 2}neighbors_recall')

            if hasattr(self, 'logger'):
                info2log = {"single_loc_recall": single_recall,
                            f"{len_neighbors ** 2}neighbors_recall": neighbors_recall,
                            f"{seq_agg_len}seq_{len_neighbors ** 2}neighbors_recall": seq_neighbors_recall}
                info2write = ''
                for key in info2log.keys():
                    info = key
                    for k, v in info2log[key].items():
                        info = info + f" @{k}:{v * 100:.2f} "
                    # self.logger.info(info)
                    info2write += info + '\n'
                if load_ckpt:
                    p_txt2write = os.path.join(os.path.dirname(self.args.ckpt2test), "recall.txt")
                    with open(p_txt2write, 'a', encoding='utf-8') as f:
                        f.write(info2write)

            if fine_loc:
                pred_nrc_list = torch.cat(pred_nrc_list, dim=0)
                gt_nrc_list = self.uav_dataloader_test.dataset.uav_nrcs_test
                error_list = eucdist_computer_rc(pred_nrc_list, gt_nrc_list)
                nrc_error = torch.mean(error_list)
                rc_error = nrc_error * self.sat_dataset.satmap_hw_max
                meter_error = rc_error * 0.5 * (abs(float(self.sat_dataset.satinfo_dict['x_resolution_m'])) + abs(
                    float(self.sat_dataset.satinfo_dict['y_resolution_m'])))
                info = f"rc_error={rc_error}; meter_error={meter_error}"
                if hasattr(self, 'logger'):
                    self.logger.info(info)
                if load_ckpt:
                    p_txt2write = os.path.join(os.path.dirname(self.args.ckpt2test), "recall.txt")
                    with open(p_txt2write, 'a', encoding='utf-8') as f:
                        f.write(info)

            # save the amt of res,todo:
            if save:
                res2save = {
                    'single_pred': pred_id_sorted,
                    'neighbors_pred': id_neighbors_1d,
                    'seq_neighbors_pred': id_neighbors_agged_1d,
                    'q_label_list': q_label_list,
                    'grid_centers': self.nrcs_grid_center.detach().cpu().numpy(),
                    'grid_n_hw': self.n_grid_hw,
                    'nrcs_grid_boundary': self.sat_dataset.nrcs_grid_boundary.detach().cpu().numpy(),
                    'pred_pdf': pred_pdf_list.detach().cpu().numpy(),
                }
                import scipy.io
                suffix = f"nmlp{self.n_grid_hw[0] * self.n_grid_hw[1]}.mat"
                scipy.io.savemat(f"{self.args.ckpt2test.replace('.pth', suffix)}", res2save)

    # def test(self,eval_on='uav',load_ckpt=False,save=False):
    #     # config the mlps according to the grids with the dataset
    #     self.cfg_mlps_w_dataset(update_args=True) if not  hasattr(self,'map_encoder') else None
    #
    #     with torch.no_grad():
    #         # config the vals
    #         args = self.args
    #         device = self.device
    #         sat_dataloader = self.sat_dataloader
    #         uav_dataloader = self.uav_dataloader_test
    #         map_encoder = self.map_encoder
    #         img_encoder = self.img_encoder
    #         eucdist_computer_rc = self.eucdist_computer_rc
    #         eucdist_computer_feat = self.eucdist_computer_feat
    #         grid_centers = self.nrcs_grid_center
    #         dist2gaussian = self.dist2gaussian
    #
    #         # load the ckpt and config if necessary
    #         if load_ckpt:
    #             checkpoint = torch.load(args.ckpt2test, map_location=device)  # map_location确保能正确加载到CPU或GPU
    #             map_encoder.load_state_dict(checkpoint['model_state_dict'])
    #         img_encoder.model.eval()
    #         map_encoder.train() if map_encoder.mlp_dropout_p>0 else map_encoder.eval()
    #
    #         # select the dataloader for test
    #         if eval_on == 'uav':
    #             dataloader = uav_dataloader
    #         else:
    #             dataloader = sat_dataloader
    #
    #         # getting the output for test
    #         q_label_list = []
    #         pred_pdf_list = []
    #         for it, data in tqdm(enumerate(dataloader)):
    #             imgs,nrcs  = data
    #             imgs,nrcs  = imgs.to(device), nrcs.to(device)
    #
    #             feats = img_encoder.model(imgs)
    #             input = feats
    #             if self.map_encoder.mlp_dropout_p>0:
    #                 batch_predictions = []
    #                 for _ in range(50):
    #                     output = map_encoder(input)
    #                     pred_pdf = torch.softmax(output, dim=-1)
    #                     batch_predictions.append(pred_pdf)
    #                 stacked_preds = torch.stack(batch_predictions, dim=1)
    #                 mean_prediction = torch.mean(stacked_preds, dim=1)
    #                 std_prediction = torch.std(stacked_preds, dim=1)
    #                 pred_pdf = std_prediction
    #                 # debug for vis
    #                 # vis_single_featmap(mean_prediction[0].reshape(self.n_grid_hw[0],self.n_grid_hw[1]),'/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/dropout_mean.png')
    #                 # vis_single_featmap(std_prediction[0].reshape(self.n_grid_hw[0],self.n_grid_hw[1]),'/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/dropout_mean.png')
    #             else:
    #                 output = map_encoder(input)
    #                 pred_pdf = torch.softmax(output, dim=-1)
    #             pred_pdf_list.append(pred_pdf)
    #
    #             nrcs2centers = eucdist_computer_rc(nrcs, grid_centers)
    #             gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers))
    #             gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)
    #             q_labels = torch.argmax(gt_pdf, dim=-1)
    #             q_label_list.append(q_labels)
    #
    #         q_label_list = torch.cat(q_label_list).cpu().numpy()
    #         pred_pdf_list = torch.cat(pred_pdf_list)
    #
    #         # compute the metrics according to the output:
    #         pred_id_sorted = torch.sort(pred_pdf_list,descending=True,dim=-1)[1]
    #         single_recall = compute_recall_by_label(q_label_list,pred_id_sorted.detach().cpu().numpy(),[1,2,3,4,5],title='single_recall')
    #
    #         len_neighbors = self.args.len_neighbors
    #         k_values = [i+1 for i in np.arange(len_neighbors**2)]
    #         id_neighbors_1d,id_neighbors_2d = compute_agged_pred_nneighbors_id(pred_pdf_list.reshape(-1,self.n_grid_hw[0],self.n_grid_hw[1]),len_neighbors,ret_2d=True)
    #         neighbors_recall = compute_recall_by_label(q_label_list, id_neighbors_1d.cpu().numpy(), k_values,title=f'{len_neighbors**2}neighbors_recall')
    #
    #         seq_agg_len = self.args.seq_agg_len
    #         pred_pdf_agged = agg_seq_pdf(pred_pdf_list,window_len=seq_agg_len)
    #         id_neighbors_agged_1d,id_neighbors_agged_2d = compute_agged_pred_nneighbors_id(pred_pdf_agged.reshape(-1,self.n_grid_hw[0],self.n_grid_hw[1]),len_neighbors,ret_2d=True)
    #         seq_neighbors_recall = compute_recall_by_label(q_label_list[-id_neighbors_agged_1d.shape[0]:], id_neighbors_agged_1d.detach().cpu().numpy(), k_values,title=f'{seq_agg_len}agged_{len_neighbors**2}neighbors_recall')
    #
    #         if hasattr(self,'logger'):
    #             info2log = {"single_loc_recall":single_recall,
    #                         f"{len_neighbors**2}neighbors_recall":neighbors_recall,
    #                         f"{seq_agg_len}seq_{len_neighbors**2}neighbors_recall":seq_neighbors_recall}
    #             for key in info2log.keys():
    #                 info = key
    #                 for k, v in info2log[key].items():
    #                     info = info + f" @{k}:{v * 100:.2f} "
    #                 self.logger.info(info)
    #
    #         # save the amt of res
    #         if save:
    #             res2save = {
    #                 'single_pred': pred_id_sorted,
    #                 'neighbors_pred': id_neighbors_1d,
    #                 'seq_neighbors_pred': id_neighbors_agged_1d,
    #                 'q_label_list': q_label_list,
    #                 'grid_centers': self.nrcs_grid_center.detach().cpu().numpy(),
    #                 'grid_n_hw': self.n_grid_hw,
    #                 'nrcs_grid_boundary': self.sat_dataset.nrcs_grid_boundary.detach().cpu().numpy(),
    #                 'pred_pdf': pred_pdf_list.detach().cpu().numpy(),
    #             }
    #             import scipy.io
    #             suffix = f"nmlp{self.n_grid_hw[0] * self.n_grid_hw[1]}.mat"
    #             scipy.io.savemat(f"{self.args.ckpt2test.replace('.pth', suffix)}", res2save)


    # def test2loc(self,eval_on='uav',load_ckpt=False):
    #     # config the mlps according to the grids with the dataset
    #     self.cfg_mlps_w_dataset(update_args=True) if not  hasattr(self,'map_encoder') else None
    #
    #     with (torch.no_grad()):
    #         # config the vals
    #         args = self.args
    #         device = self.device
    #         sat_dataloader = self.sat_dataloader
    #         uav_dataloader = self.uav_dataloader_test
    #         map_encoder = self.map_encoder
    #         img_encoder = self.img_encoder
    #         eucdist_computer_feat = self.eucdist_computer_feat
    #         eucdist_computer_rc = self.eucdist_computer_rc
    #         grid_centers = self.nrcs_grid_center
    #         dist2gaussian = self.dist2gaussian
    #         resize_transform = T.Resize(self.args.imgsize2net)
    #
    #         if load_ckpt:
    #             checkpoint = torch.load(args.ckpt2test, map_location=device)  # map_location确保能正确加载到CPU或GPU
    #             map_encoder.load_state_dict(checkpoint['model_state_dict'])
    #         img_encoder.model.eval()
    #         map_encoder.train() if map_encoder.mlp_dropout_p>0 else map_encoder.eval()
    #
    #         #test on uav_imgs
    #         if eval_on == 'uav':
    #             dataloader = uav_dataloader
    #         else:
    #             dataloader = sat_dataloader
    #
    #         pred_nrc_list = []
    #         gt_in_neighbors_list = []
    #         q_label_list, pred_classid_list = [],[]
    #         id_neighbors_list = []
    #         pred_pdf_list = []
    #         for it, data in tqdm(enumerate(dataloader)):
    #             imgs,nrcs  = data
    #             imgs,nrcs  = imgs.to(device), nrcs.to(device)
    #
    #             feats_q = img_encoder.model(imgs)
    #             input = feats_q
    #             if self.map_encoder.mlp_dropout_p>0:
    #                 batch_predictions = []
    #                 for _ in range(50):
    #                     output = map_encoder(input)
    #                     pred_pdf = torch.softmax(output, dim=-1)
    #                     batch_predictions.append(pred_pdf)
    #                 stacked_preds = torch.stack(batch_predictions, dim=1)
    #                 mean_prediction = torch.mean(stacked_preds, dim=1)
    #                 std_prediction = torch.std(stacked_preds, dim=1)
    #                 pred_pdf = std_prediction
    #             else:
    #                 output = map_encoder(input)
    #                 pred_pdf = torch.softmax(output, dim=-1)
    #
    #             nrcs2centers = eucdist_computer_rc(nrcs, grid_centers)
    #             gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers))
    #             gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)
    #
    #             q_labels = torch.argmax(gt_pdf, dim=-1)
    #             q_label_list.append(q_labels)
    #             orted_values, sorted_indices = torch.sort(pred_pdf, descending=True)
    #             pred_classid_list.append(sorted_indices)
    #             pred_pdf_list.append(pred_pdf)
    #
    #             len_neighbors = self.args.len_neighbors
    #             id_neighbors_flat,id_neighbors_2d = compute_agged_pred_nneighbors_id(pred_pdf.reshape(-1,self.n_grid_hw[0],self.n_grid_hw[1]),len_neighbors,ret_2d=True)
    #             id_neighbors_list.append(id_neighbors_flat)
    #             nrcs_topleft = self.sat_dataset.nrcs_grid_boundary[id_neighbors_2d[:,0,0].cpu(),id_neighbors_2d[:,0,1].cpu()]
    #             nrcs_rightbottom = nrcs_topleft + torch.stack([self.sat_dataset.gird_cell_h,self.sat_dataset.gird_cell_w])[None,...]*len_neighbors
    #
    #             nrc_topleft_previous = None
    #             for i, nrc_topleft in enumerate(nrcs_topleft):
    #                 # --- 1. 判断当前坐标是否需要处理 ---
    #                 # 条件：是第一个元素，或者当前元素与上一个已处理的元素不同
    #                 should_process = (i == 0) or (torch.abs(nrc_topleft - nrc_topleft_previous).sum() > 1e-5)
    #
    #                 # --- 2. 如果需要处理，则执行特征提取 ---
    #                 if should_process:
    #                     # 这个代码块现在是唯一的，不再重复
    #                     sat_tiles_tensor, nrcs_grid_center = self.sat_dataset.sample_sats_in_rect(
    #                         nrc_topleft,  # 使用当前的 nrc_topleft
    #                         nrcs_rightbottom[i],
    #                         satimgsize2crop=self.args.satimgsize2crop,
    #                         type2clip='tensor'
    #                     )
    #                     sat_tiles_tensor = resize_transform(sat_tiles_tensor.reshape(-1, *sat_tiles_tensor.shape[2:]))
    #
    #                     split_size = 256
    #                     feat_gallery = []
    #                     print("开始为新区域分批提取特征...")
    #                     for tiles_batch_cpu in tqdm(torch.split(sat_tiles_tensor, split_size, dim=0),
    #                                                 desc="Extracting Features"):
    #                         tiles_batch_gpu = tiles_batch_cpu.to(self.device)
    #                         with torch.no_grad():  # 在推理时使用 no_grad() 以节省显存和加速
    #                             output = self.img_encoder.model(tiles_batch_gpu)
    #
    #                         feat_g = output.detach().cpu()
    #                         feat_g = feat_g.view(feat_g.shape[0], -1) if len(feat_g.shape) > 2 else feat_g
    #                         feat_gallery.append(feat_g)
    #
    #                     feat_gallery = torch.cat(feat_gallery, dim=0)
    #                     print("特征提取完成。")
    #                     # 更新“上一个已处理”的坐标
    #                     nrc_topleft_previous = nrc_topleft
    #
    #                 # 计算定位的概率分布
    #                 feat_dist = eucdist_computer_feat(feats_q[i][None,:].cpu(), feat_gallery)
    #                 dist2prob_scale = 2.0
    #                 nrc_distb = torch.exp(-dist2prob_scale * feat_dist).reshape(*nrcs_grid_center.shape[:2])
    #                 nrc_distb_normed = nrc_distb / nrc_distb.sum()
    #                 # 计算精细定位结果
    #                 pred_r, pred_c = torch.where(nrc_distb_normed == nrc_distb_normed.max())
    #                 pred_nrc_in_grid = torch.tensor([pred_r/nrc_distb.shape[0],pred_c/nrc_distb.shape[1]])*torch.tensor([self.sat_dataset.gird_cell_h,self.sat_dataset.gird_cell_w])*len_neighbors
    #                 pred_nrc = pred_nrc_in_grid+nrcs_topleft[i]
    #                 pred_nrc_list.append(pred_nrc)
    #
    #                 # debug 验证定位结果, #todo: checkout the uav_nrc
    #                 from vis_featmap import vis_single_featmap,add_pts2map
    #                 dir2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_zuchwil_3/exp_vis'
    #                 p_uavimg = self.uav_dataset_test.uavimg_paths_test[i]
    #                 uavimg_name = p_uavimg.split('/')[-1].split('.')[0][-8:]
    #                 p2save_xyprob = f'{dir2save}/xy_prob_{uavimg_name}.png'
    #                 fig,ax = vis_single_featmap(nrc_distb_normed,p2save=p2save_xyprob,return_handles=True)
    #                 gt_nrc_in_grid = nrcs[i].cpu()-nrcs_topleft[i]
    #                 gt_rc = gt_nrc_in_grid /len_neighbors / torch.tensor([self.sat_dataset.gird_cell_h,self.sat_dataset.gird_cell_w]) * torch.tensor([nrc_distb_normed.shape[0],nrc_distb_normed.shape[1]])
    #                 pred_rc = torch.tensor([pred_r,pred_c])
    #                 add_pts2map(fig,ax,torch.stack([gt_rc,pred_rc]),labels=['gt','pred'],p2save=p2save_xyprob)
    #                 from matplotlib import pyplot as plt
    #                 from PIL import Image
    #                 uavimg = Image.open(p_uavimg)
    #                 plt.imshow(uavimg)
    #                 plt.savefig(f'{dir2save}/uavimg_{uavimg_name}.png')
    #                 plt.close()
    #                 satimg_rect = self.sat_dataset.crop_rect_satimg(nrc_topleft=nrcs_topleft[i],nrc_rightbottom=nrcs_rightbottom[i], type='np')
    #                 plt.imshow(satimg_rect)
    #                 plt.savefig(f'{dir2save}/rect_satimg_{uavimg_name}.png')
    #                 plt.close()
    #                 uav_nrc =  self.uav_dataset_test.uav_nrcs_test[i]
    #                 satimg = self.sat_dataset.crop_satimg_by_nrc(uav_nrc,type='np')
    #                 plt.imshow(satimg)
    #                 plt.savefig(f'{dir2save}/satimg_{uavimg_name}.png')
    #                 plt.close()
    #
    #             gt_in_neighbors=[]
    #             for i,label in enumerate(q_labels):
    #                 if label in id_neighbors_flat[i]:
    #                     gt_in_neighbors.append(True)
    #                 else:
    #                     gt_in_neighbors.append(False)
    #             gt_in_neighbors = np.stack(gt_in_neighbors)
    #             gt_in_neighbors_list.append(gt_in_neighbors)
    #
    #
    #         q_label_list = torch.cat(q_label_list).cpu().numpy()
    #         pred_classid_list = torch.cat(pred_classid_list).cpu().numpy()
    #         single_loc_recall = compute_recall_by_label(q_label_list,pred_classid_list,[1,2,3,4,5],title='single_loc_recall')
    #         id_neighbors_list = torch.cat(id_neighbors_list).cpu().numpy()
    #         single_loc_4neighbor_recall = compute_recall_by_label(q_label_list, id_neighbors_list, [1, 2, 3, 4],title='single_loc_4neighbor_recall')
    #         pred_pdf_list = torch.cat(pred_pdf_list)
    #         pred_seq_agged = agg_seq_pdf(pred_pdf_list)
    #         id_neighbors_flat = compute_agged_pred_4neighbors_id(pred_seq_agged.reshape(-1,self.n_grid_hw[0],self.n_grid_hw[1])).cpu().numpy()
    #         seq_agged_loc_4neighbor_recall = compute_recall_by_label(q_label_list[-id_neighbors_flat.shape[0]:], id_neighbors_flat, [1, 2, 3, 4],title='seq_agged_loc_4neighbor_recall')
    #
    #         if hasattr(self,'logger'):
    #             info2log = {"single_loc_recall":single_loc_recall,"single_loc_4neighbor_recall":single_loc_4neighbor_recall,"seq_agged_loc_4neighbor_recall":seq_agged_loc_4neighbor_recall}
    #             for key in info2log.keys():
    #                 info = key
    #                 for k, v in info2log[key].items():
    #                     info = info + f" @{k}:{v * 100:.2f} "
    #                 self.logger.info(info)

            #save the amt of res:
            # res2save = {
            #     'single_pred':pred_classid_list,
            #     '4neighbors_pred':id_neighbors_list,
            #     'seq_agged_4neighbor_pred':id_neighbors_flat,
            #     'q_label_list':q_label_list,
            #     'grid_centers':self.nrcs_grid_center.detach().cpu().numpy(),
            #     'grid_n_hw':self.n_grid_hw,
            #     'nrcs_grid_boundary':self.sat_dataset.nrcs_grid_boundary.detach().cpu().numpy(),
            # }
            # import scipy.io
            # suffix = f"nmlp{self.n_grid_hw[0]*self.n_grid_hw[1]}.mat"
            # scipy.io.savemat(f"{self.args.ckpt2test.replace('.pth', suffix)}",res2save)
            # res = scipy.io.loadmat(f"{self.args.ckpt2test.replace('.pth', suffix)}")

    # def test_seq(self):
    #     import scipy.io
    #     suffix = f"nmlp{self.n_grid_hw[0] * self.n_grid_hw[1]}.mat"
    #     res = scipy.io.loadmat(f"{self.args.ckpt2test.replace('.pth', suffix)}")
    #     grid_n_hw = res['grid_n_hw'].squeeze()
    #     pred_pdf = torch.tensor(res['pred_pdf']).reshape(-1,*grid_n_hw)
    #     pred_label_1d = res['single_pred'].squeeze()
    #     pred_label_2d = np.stack([pred_label_1d//grid_n_hw[1],pred_label_1d%grid_n_hw[1]],axis=-1)
    #     gt_label_1d = res['q_label_list'].squeeze()
    #     gt_label_2d = np.stack([gt_label_1d//grid_n_hw[1],gt_label_1d%grid_n_hw[1]],axis=-1)
    #     from vis_featmap import vis_multi_featmap,add_gt2maps
    #     p2save = '/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/perd_pdf[18:27].png'
    #     fig,axs=vis_multi_featmap(pred_pdf[18:27],p2save=p2save,return_handles=True)
    #     add_gt2maps(fig,axs,gt_label_2d[18:27],p2save=p2save)

    # def compute_recall(self, pred_pdf, gt_pdf, gt_nrcs, grid_centers, grid_cell_radius):
    #     id_gt = torch.argmax(gt_pdf, dim=-1, keepdim=False)
    #     id_pred = torch.argmax(pred_pdf, dim=-1, keepdim=False)
    #     recall_1 = (id_pred == id_gt).sum() / id_pred.shape[0]
    #     pred_rcs = grid_centers[id_pred]
    #     dist_rel2radius = torch.norm(pred_rcs - gt_nrcs, dim=-1) / grid_cell_radius
    #     dist_rel2radius_recall_1 = dist_rel2radius.mean()
    #     return recall_1,dist_rel2radius_recall_1

if __name__ == "__main__":
    tranier = Trainer()
    tranier.train()
    # tranier.test(eval_on='uav',load_ckpt=True, fine_loc=False)
    # tranier.test_seq()







