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

from vis_featmap import vis_multi_featmap,vis_single_featmap
from eval_recall_fm_salad import compute_recall_by_label
from loc_utils import agg_seq_pdf,compute_agged_pred_4neighbors_id,find_4neighbors_topleft,compute_agged_pred_nneighbors_id,find_nneighbors_topleft,compute_l2_dist


def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")
    parser.add_argument('--exp_name', default='debug_zurich', type=str)
    parser.add_argument('--exps_dir', default='/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps', type=str)
    parser.add_argument('--p_satinfo_json',
                        default='/home/data/zwk/data_uavimgs_wingtra/Zurich/SWISSIMAGE2022_cover_uavimgs_proj2056_edgepix1024_res02m.json',
                        type=str, help='training sat dir path')
    parser.add_argument('--p_uavinfo_json',
                        default='/home/data/zwk/data_uavimgs_wingtra/Zurich/uavimgs_info.json',
                        type=str, help='training uav dir path')
    parser.add_argument('--p_img_encoder_cfg',
                        default='/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug_zurich_11/opts_wingtra.yaml',
                        type=str, help='training uav dir path')
    parser.add_argument('--p_img_encoder_ckpt',
                        default='/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug_zurich_11/epoch008.pth',
                        type=str, help='training uav dir path')
    parser.add_argument('--imgsize2net',default = 224, type=int, help='the img size to net')
    parser.add_argument('--satimgsize2crop',default = 614, type=int, help='the satimg cliped from tif')
    parser.add_argument('--grid_scale_on_satimgsize2crop', default=1.5, type=int, help='be used to determine the size of each classified area')
    parser.add_argument('--n_neighbors', default = 4, type=int, help='4 or 9 or 25...')
    parser.add_argument('--seq_agg_len', default= 3, type=int, help='2,3,4...')
    parser.add_argument('--batchsize_sat', default=32, type=int, help='batchsize')
    parser.add_argument('--batchsize_uav', default=32, type=int, help='batchsize')
    parser.add_argument("--n_epoch", nargs="?", type=int, default=100, help="Number of training steps")
    parser.add_argument('--num_worker', default=8, type=int)
    parser.add_argument('--tensorboard', action='store_true', default = False)
    parser.add_argument('--ckpt2test', default="/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_zuchwil_1/epoch_43.pth", type=str, help='path for testing') # for testing
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
            p_uavinfo_json=self.args.p_uavinfo_json,
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
        self.n_grid_hw = torch.tensor((self.sat_dataset.n_grid_h, self.sat_dataset.n_grid_w))
        self.gird_cell_hw = torch.tensor((self.sat_dataset.gird_cell_h,self.sat_dataset.gird_cell_w))
        self.dist2gspdf = torch.distributions.Normal(loc=0,scale=self.sat_dataset.grid_cell_radius * 0.4)  # scale=sigma, 2sigma that contains 99% pdf = grid_cell_radius
        self.eucdist_computer_feat = LpDistance(normalize_embeddings=True)
        self.eucdist_computer_rc = LpDistance(normalize_embeddings=False)

        # debug,vis the gird
        # from vis_loc_res import vis_rcs_on_tif,draw_nrc_grid_on_image
        # pts = torch.cat([self.nrcs_grid_center.detach().cpu(), self.sat_dataset.nrcs_grid_boundary.reshape(-1, 2)]).numpy()
        # vis_rcs_on_tif(np.array(self.sat_dataset.satmap), self.nrcs_grid_center.detach().cpu().numpy(),
        #                p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/grid_centers.png',
        #                color='red')
        # vis_rcs_on_tif(np.array(self.sat_dataset.satmap),
        #                self.sat_dataset.nrcs_grid_boundary.reshape(-1, 2).detach().cpu().numpy(),
        #                p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_zuchwil/nrcs_grid_boundary_point.png',
        #                color='blue')
        # draw_nrc_grid_on_image(np.array(self.sat_dataset.satmap),
        #                    self.sat_dataset.nrcs_grid_boundary.detach().cpu().numpy(),
        #                     p2save = '/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_zuchwil/nrcs_grid_boundary_line.png',
        #                    )
        # vis_rcs_on_tif(np.array(self.sat_dataset.satmap), pts,
        #                p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/grid_cneter&boundary',
        #                color='blue')

        from get_pos_encoder import PositionalEncoder
        pos_encoder = PositionalEncoder(multires=6, input_dims=2)
        self.pos_encoder = pos_encoder
        from get_mlps_classify import create_mlp, init_weights, MultiMLP
        input_dim = self.img_encoder.get_output_dim() + pos_encoder.out_dim
        # input_dim = 768
        mlp_dropout_p = 0.
        self.map_encoder = MultiMLP(self.nrcs_grid_center.shape[0], input_dim=input_dim,
                                    mlp_hidden_dims=[256, 128, 64],  # [ 128, 32, 8]
                                    mlp_activation_fn=nn.LeakyReLU, mlp_norm_type='layer',
                                    mlp_dropout_p=mlp_dropout_p,
                                    mlp_init_method='kaiming', mlp_init_nonlinearity='leaky_relu',
                                    device=self.device)
        self.map_encoder.mlp_dropout_p = mlp_dropout_p


    def train(self):
        # config the mlps according to the grids with the dataset
        self.cfg_mlps_w_dataset()

        args = self.args
        device = self.device
        map_encoder = self.map_encoder
        img_encoder = self.img_encoder
        # eucdist_computer_feat = self.eucdist_computer_feat
        eucdist_computer_rc = self.eucdist_computer_rc
        grid_centers = self.nrcs_grid_center
        dist2gspdf = self.dist2gspdf

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

        epoch_begin = 0
        if args.ckpt2train != "":
            checkpoint = torch.load(args.ckpt2train, map_location=device)  # map_location确保能正确加载到CPU或GPU
            map_encoder.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_begin = checkpoint['epoch']

        #config for recording exp & backup files:
        from tool.utils import get_unique_exp_dir,copyfiles2checkpoints_map_learner
        self.args.exp_name = get_unique_exp_dir(self.args.exps_dir,self.args.exp_name)
        expdir2save = "{}/{}".format(self.args.exps_dir,self.args.exp_name)
        if not os.path.exists(expdir2save):
            os.mkdir(expdir2save)
        self.writer = SummaryWriter(f"{expdir2save}/{args.exp_name}/train_tensorboard.log") if args.tensorboard else None
        from tool.utils import  get_logger
        self.logger = get_logger("{}/{}/train.log".format(self.args.exps_dir,self.args.exp_name),'trainer_logger')
        copyfiles2checkpoints_map_learner( self.args )

        from facol_loss import SoftFocalLoss
        criterion_logits = SoftFocalLoss(gamma=6,reduction='sum')
        # criterion_logits = nn.BCEWithLogitsLoss(reduction='mean')
        for epoch in torch.range(epoch_begin,args.n_epoch,dtype=torch.uint8):
            for it, data in tqdm(enumerate(self.sat_dataloader)):
                imgs, nrcs = data
                uav_imgs,uav_nrcs = next(iter(self.uav_dataloader_train))
                nrcs = torch.concatenate([nrcs,uav_nrcs],dim=0).to(device)
                imgs = torch.concatenate([imgs,uav_imgs],dim=0).to(device)
                img_feats = img_encoder.model(imgs)

                #compute the local nrcs
                nrcs2centers = compute_l2_dist(nrcs, self.nrcs_grid_center)
                gtnrcs_gid_center = self.nrcs_grid_center[torch.argmin(nrcs2centers, dim=1)]
                gtnrcs_in_grid = nrcs - gtnrcs_gid_center
                gtnrcs_in_grid = gtnrcs_in_grid / self.gird_cell_hw[None, :].to(self.device)
                gtnrcs_in_grid_pe = self.pos_encoder(gtnrcs_in_grid)

                input = torch.cat([img_feats, gtnrcs_in_grid_pe], dim=-1)
                output = map_encoder.model(input)

                # generate gtpdf for the outputs from multimlps
                dist_mat = compute_l2_dist(nrcs, nrcs)
                gt_pdf = torch.exp(self.dist2gspdf.log_prob(dist_mat))
                gt_pdf = gt_pdf / torch.max(gt_pdf,dim=-1)[0][:,None]

                #generate gtpdf for the outputs from multimlps
                # nrcs_global = gtnrcs_in_grid*self.gird_cell_hw[None, :].to(self.device)
                # nrcs_global = nrcs_global.unsqueeze(1)+self.nrcs_grid_center.unsqueeze(0)
                # nrcs_global_flatten = nrcs_global.reshape(-1,2)
                # dist_mat = compute_l2_dist(nrcs,nrcs_global_flatten)
                # gt_pdf = torch.exp(self.dist2gspdf.log_prob(dist_mat))
                # gt_pdf = gt_pdf / torch.max(gt_pdf,dim=-1)[0][:,None]

                # debug, vis the gt_pdf
                # import matplotlib.pyplot as plt
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # # ax.plot_trisurf(nrcs_global_flatten.reshape(-1, 2)[:,1].detach().cpu().numpy(), nrcs_global_flatten.reshape(-1, 2)[:,0].detach().cpu().numpy(), gt_pdf[2].detach().cpu().numpy(), cmap='viridis', edgecolor='none')
                # ax.scatter(nrcs_global_flatten[:, 1].detach().cpu().numpy(),nrcs_global_flatten[:, 0].detach().cpu().numpy(), gt_pdf[1].detach().cpu().numpy(),s=2)
                # # ax.bar3d(nrcs[:,1].detach().cpu().numpy(), nrcs[:,0].detach().cpu().numpy(), np.zeros_like(gt_pdf[0].detach().cpu().numpy()).ravel(), 1, 1, gt_pdf[0].detach().cpu().numpy(), shade=True)
                # ax.set_xlabel("X")
                # ax.set_ylabel("Y")
                # plt.savefig(f"/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/vis_occloc/pdf.png")
                # for img_id,img_feat in enumerate(img_feats):
                #     # loc 参数是均值(mean), scale 参数是标准差(std)
                #     normal_dist = torch.distributions.Normal(loc=gtnrcs_in_grid[img_id], scale=0.2)
                #     samples = normal_dist.sample((1024,))
                #     samples = samples[(samples[:,0]<0.5) * (samples[:,0]>-0.5) * (samples[:,1]>-0.5) *(samples[:,1]<0.5)]
                #
                #
                #     input = torch.cat([img_feat[None,:].expand(gtnrcs_in_grid_pe.shape[0],-1),gtnrcs_in_grid_pe],dim=-1)
                #     output = map_encoder(input)
                loss = criterion_logits(output.flatten(), gt_pdf[img_id])

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                print(f"loss={loss:.4f}")
                self.writer.add_scalar('loss', loss, it) if self.writer is not None else None

            # recall1,recall1_rel_dist = self.test(eval_on='uav',load_ckpt=False)
            self.test(eval_on='uav', load_ckpt=False)
            self.map_encoder.train()
            self.img_encoder.model.train()
            # self.writer.add_scalar('recall1', recall1, epoch) if self.writer is not None else None
            # self.writer.add_scalar('recall1_rel_dist', recall1_rel_dist, epoch) if self.writer is not None else None
            self.logger.info(f"ep={epoch} finished ") if hasattr(self,'logger') else None

            # if epoch % 5 == 0:
            dir2save = f"{self.args.exps_dir}/{self.args.exp_name}"
            torch.save({
                'epoch': epoch,
                'model_state_dict': map_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,  # 可选：保存最后一个 batch 的损失值
            }, os.path.join(dir2save, f"epoch_{epoch}.pth"))



    def test(self,eval_on='uav',load_ckpt=False,save=False):
        with torch.no_grad():
            args = self.args
            device = self.device
            sat_dataloader = self.sat_dataloader
            uav_dataloader = self.uav_dataloader_test
            # pos_encoder = self.pos_encoder
            map_encoder = self.map_encoder
            img_encoder = self.img_encoder
            # eucdist_computer_feat = self.eucdist_computer_feat
            eucdist_computer_rc = self.eucdist_computer_rc
            grid_centers = self.nrcs_grid_center
            dist2gspdf = self.dist2gspdf

            if load_ckpt:
                checkpoint = torch.load(args.ckpt2test, map_location=device)  # map_location确保能正确加载到CPU或GPU
                map_encoder.load_state_dict(checkpoint['model_state_dict'])
            img_encoder.model.eval()
            map_encoder.train() if map_encoder.mlp_dropout_p>0 else map_encoder.eval()

            #test on uav_imgs
            if eval_on == 'uav':
                dataloader = uav_dataloader
            else:
                dataloader = sat_dataloader

            q_label_list = []
            pred_pdf_list = []
            for it, data in tqdm(enumerate(dataloader)):
                imgs,nrcs  = data
                imgs,nrcs  = imgs.to(device), nrcs.to(device)

                feats = img_encoder.model(imgs)
                input = feats
                if self.map_encoder.mlp_dropout_p>0:
                    batch_predictions = []
                    for _ in range(50):
                        output = map_encoder(input)
                        pred_pdf = torch.softmax(output, dim=-1)
                        batch_predictions.append(pred_pdf)
                    stacked_preds = torch.stack(batch_predictions, dim=1)
                    mean_prediction = torch.mean(stacked_preds, dim=1)
                    std_prediction = torch.std(stacked_preds, dim=1)
                    pred_pdf = std_prediction
                    # debug for vis
                    # vis_single_featmap(mean_prediction[0].reshape(self.n_grid_hw[0],self.n_grid_hw[1]),'/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/dropout_mean.png')
                    # vis_single_featmap(std_prediction[0].reshape(self.n_grid_hw[0],self.n_grid_hw[1]),'/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/dropout_mean.png')
                else:
                    output = map_encoder(input)
                    pred_pdf = torch.softmax(output, dim=-1)

                nrcs2centers = eucdist_computer_rc(nrcs, grid_centers)
                gt_pdf = torch.exp(dist2gspdf.log_prob(nrcs2centers))
                gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)

                q_labels = torch.argmax(gt_pdf, dim=-1)
                q_label_list.append(q_labels)
                pred_pdf_list.append(pred_pdf)

            q_label_list = torch.cat(q_label_list).cpu().numpy()
            pred_pdf_list = torch.cat(pred_pdf_list)

            pred_id_sorted = torch.sort(pred_pdf_list,descending=True,dim=-1)[1]
            single_recall = compute_recall_by_label(q_label_list,pred_id_sorted.detach().cpu().numpy(),[1,2,3,4,5],title='single_recall')

            len_neighbors = 2
            k_values = [i+1 for i in np.arange(len_neighbors**2)]
            id_neighbors_1d,id_neighbors_2d = compute_agged_pred_nneighbors_id(pred_pdf_list.reshape(-1,self.n_grid_hw[0],self.n_grid_hw[1]),len_neighbors,ret_2d=True)
            neighbors_recall = compute_recall_by_label(q_label_list, id_neighbors_1d.cpu().numpy(), k_values,title=f'{len_neighbors**2}neighbors_recall')

            seq_window_len = 3
            pred_pdf_agged = agg_seq_pdf(pred_pdf_list,window_len=seq_window_len)
            id_neighbors_agged_1d,id_neighbors_agged_2d = compute_agged_pred_nneighbors_id(pred_pdf_agged.reshape(-1,self.n_grid_hw[0],self.n_grid_hw[1]),len_neighbors,ret_2d=True)
            seq_neighbors_recall = compute_recall_by_label(q_label_list[-id_neighbors_agged_1d.shape[0]:], id_neighbors_agged_1d.detach().cpu().numpy(), k_values,title=f'{seq_window_len}agged_{len_neighbors**2}neighbors_recall')

            if hasattr(self,'logger'):
                info2log = {"single_loc_recall":single_recall,
                            f"{len_neighbors**2}neighbors_recall":neighbors_recall,
                            f"{seq_window_len}seq_{len_neighbors**2}neighbors_recall":seq_neighbors_recall}
                for key in info2log.keys():
                    info = key
                    for k, v in info2log[key].items():
                        info = info + f" @{k}:{v * 100:.2f} "
                    self.logger.info(info)

            #save the amt of res:
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


    def test2loc(self,eval_on='uav',load_ckpt=False):
        # config the mlps according to the grids with the dataset
        self.cfg_mlps_w_dataset(update_args=True)

        with (torch.no_grad()):
            args = self.args
            device = self.device
            sat_dataloader = self.sat_dataloader
            uav_dataloader = self.uav_dataloader_test
            # pos_encoder = self.pos_encoder
            map_encoder = self.map_encoder
            img_encoder = self.img_encoder
            eucdist_computer_feat = self.eucdist_computer_feat
            eucdist_computer_rc = self.eucdist_computer_rc
            grid_centers = self.nrcs_grid_center
            dist2gspdf = self.dist2gspdf
            resize_transform = T.Resize(self.args.imgsize2net)

            if load_ckpt:
                checkpoint = torch.load(args.ckpt2test, map_location=device)  # map_location确保能正确加载到CPU或GPU
                map_encoder.load_state_dict(checkpoint['model_state_dict'])
            img_encoder.model.eval()
            map_encoder.train() if map_encoder.mlp_dropout_p>0 else map_encoder.eval()

            #test on uav_imgs
            if eval_on == 'uav':
                dataloader = uav_dataloader
            else:
                dataloader = sat_dataloader

            pred_nrc_list = []
            gt_in_neighbors_list = []
            q_label_list, pred_classid_list = [],[]
            id_4neighbors_list = []
            pred_pdf_list = []
            for it, data in tqdm(enumerate(dataloader)):
                imgs,nrcs  = data
                imgs,nrcs  = imgs.to(device), nrcs.to(device)

                feats_q = img_encoder.model(imgs)
                input = feats_q
                if self.map_encoder.mlp_dropout_p>0:
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

                nrcs2centers = eucdist_computer_rc(nrcs, grid_centers)
                gt_pdf = torch.exp(dist2gspdf.log_prob(nrcs2centers))
                gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)

                q_labels = torch.argmax(gt_pdf, dim=-1)
                q_label_list.append(q_labels)
                orted_values, sorted_indices = torch.sort(pred_pdf, descending=True)
                pred_classid_list.append(sorted_indices)
                pred_pdf_list.append(pred_pdf)

                len_neighbors = 2
                id_4neighbors_flat,id_4neighbors_2d = compute_agged_pred_nneighbors_id(pred_pdf.reshape(-1,self.n_grid_hw[0],self.n_grid_hw[1]),len_neighbors,ret_2d=True)
                id_4neighbors_list.append(id_4neighbors_flat)
                nrcs_topleft = self.sat_dataset.nrcs_grid_boundary[id_4neighbors_2d[:,0,0].cpu(),id_4neighbors_2d[:,0,1].cpu()]
                nrcs_rightbottom = nrcs_topleft + torch.stack([self.sat_dataset.gird_cell_h,self.sat_dataset.gird_cell_w])[None,...]*len_neighbors

                nrc_topleft_previous = None
                for i, nrc_topleft in enumerate(nrcs_topleft):
                    # --- 1. 判断当前坐标是否需要处理 ---
                    # 条件：是第一个元素，或者当前元素与上一个已处理的元素不同
                    should_process = (i == 0) or (torch.abs(nrc_topleft - nrc_topleft_previous).sum() > 1e-5)

                    # --- 2. 如果需要处理，则执行特征提取 ---
                    if should_process:
                        # 这个代码块现在是唯一的，不再重复
                        sat_tiles_tensor, nrcs_grid_center = self.sat_dataset.sample_sats_in_rect(
                            nrc_topleft,  # 使用当前的 nrc_topleft
                            nrcs_rightbottom[i],
                            satimgsize2crop=self.args.satimgsize2crop,
                            type2clip='tensor'
                        )
                        sat_tiles_tensor = resize_transform(sat_tiles_tensor.reshape(-1, *sat_tiles_tensor.shape[2:]))

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
                    feat_dist = eucdist_computer_feat(feats_q[i][None,:].cpu(), feat_gallery)
                    dist2prob_scale = 2.0
                    nrc_distb = torch.exp(-dist2prob_scale * feat_dist).reshape(*nrcs_grid_center.shape[:2])
                    nrc_distb_normed = nrc_distb / nrc_distb.sum()
                    # 计算精细定位结果
                    pred_r, pred_c = torch.where(nrc_distb_normed == nrc_distb_normed.max())
                    pred_nrc_in_grid = torch.tensor([pred_r/nrc_distb.shape[0],pred_c/nrc_distb.shape[1]])*torch.tensor([self.sat_dataset.gird_cell_h,self.sat_dataset.gird_cell_w])*len_neighbors
                    pred_nrc = pred_nrc_in_grid+nrcs_topleft[i]
                    pred_nrc_list.append(pred_nrc)

                    # debug 验证定位结果, #todo: checkout the uav_nrc
                    # from vis_featmap import vis_single_featmap,add_pts2map
                    # dir2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_zuchwil_2/res_vis'
                    # p_uavimg = self.uav_dataset_test.uavimg_paths_test[i]
                    # uavimg_name = p_uavimg.split('/')[-1].split('.')[0][-8:]
                    # p2save_xyprob = f'{dir2save}/xy_prob_{uavimg_name}.png'
                    # fig,ax = vis_single_featmap(nrc_distb_normed,p2save=p2save_xyprob,return_handles=True)
                    # gt_nrc_in_grid = nrcs[i].cpu()-nrcs_topleft[i]
                    # gt_rc = gt_nrc_in_grid /len_neighbors / torch.tensor([self.sat_dataset.gird_cell_h,self.sat_dataset.gird_cell_w]) * torch.tensor([nrc_distb_normed.shape[0],nrc_distb_normed.shape[1]])
                    # pred_rc = torch.tensor([pred_r,pred_c])
                    # add_pts2map(fig,ax,torch.stack([gt_rc,pred_rc]),labels=['gt','pred'],p2save=p2save_xyprob)
                    # from matplotlib import pyplot as plt
                    # from PIL import Image
                    # uavimg = Image.open(p_uavimg)
                    # plt.imshow(uavimg)
                    # plt.savefig(f'{dir2save}/uavimg_{uavimg_name}.png')
                    # plt.close()
                    # satimg_rect = self.sat_dataset.crop_rect_satimg(nrc_topleft=nrcs_topleft[i],nrc_rightbottom=nrcs_rightbottom[i], type='np')
                    # plt.imshow(satimg_rect)
                    # plt.savefig(f'{dir2save}/rect_satimg_{uavimg_name}.png')
                    # plt.close()
                    # uav_nrc =  self.uav_dataset_test.uav_nrcs_test[i]
                    # satimg = self.sat_dataset.crop_satimg_by_nrc(uav_nrc,type='np')
                    # plt.imshow(satimg)
                    # plt.savefig(f'{dir2save}/satimg_{uavimg_name}.png')
                    # plt.close()

                gt_in_neighbors=[]
                for i,label in enumerate(q_labels):
                    if label in id_4neighbors_flat[i]:
                        gt_in_neighbors.append(True)
                    else:
                        gt_in_neighbors.append(False)
                gt_in_neighbors = np.stack(gt_in_neighbors)
                gt_in_neighbors_list.append(gt_in_neighbors)


            q_label_list = torch.cat(q_label_list).cpu().numpy()
            pred_classid_list = torch.cat(pred_classid_list).cpu().numpy()
            single_loc_recall = compute_recall_by_label(q_label_list,pred_classid_list,[1,2,3,4,5],title='single_loc_recall')
            id_4neighbors_list = torch.cat(id_4neighbors_list).cpu().numpy()
            single_loc_4neighbor_recall = compute_recall_by_label(q_label_list, id_4neighbors_list, [1, 2, 3, 4],title='single_loc_4neighbor_recall')
            pred_pdf_list = torch.cat(pred_pdf_list)
            pred_seq_agged = agg_seq_pdf(pred_pdf_list)
            id_4neighbors_flat = compute_agged_pred_4neighbors_id(pred_seq_agged.reshape(-1,self.n_grid_hw[0],self.n_grid_hw[1])).cpu().numpy()
            seq_agged_loc_4neighbor_recall = compute_recall_by_label(q_label_list[-id_4neighbors_flat.shape[0]:], id_4neighbors_flat, [1, 2, 3, 4],title='seq_agged_loc_4neighbor_recall')

            if hasattr(self,'logger'):
                info2log = {"single_loc_recall":single_loc_recall,"single_loc_4neighbor_recall":single_loc_4neighbor_recall,"seq_agged_loc_4neighbor_recall":seq_agged_loc_4neighbor_recall}
                for key in info2log.keys():
                    info = key
                    for k, v in info2log[key].items():
                        info = info + f" @{k}:{v * 100:.2f} "
                    self.logger.info(info)

            #save the amt of res:
            # res2save = {
            #     'single_pred':pred_classid_list,
            #     '4neighbors_pred':id_4neighbors_list,
            #     'seq_agged_4neighbor_pred':id_4neighbors_flat,
            #     'q_label_list':q_label_list,
            #     'grid_centers':self.nrcs_grid_center.detach().cpu().numpy(),
            #     'grid_n_hw':self.n_grid_hw,
            #     'nrcs_grid_boundary':self.sat_dataset.nrcs_grid_boundary.detach().cpu().numpy(),
            # }
            # import scipy.io
            # suffix = f"nmlp{self.n_grid_hw[0]*self.n_grid_hw[1]}.mat"
            # scipy.io.savemat(f"{self.args.ckpt2test.replace('.pth', suffix)}",res2save)
            # res = scipy.io.loadmat(f"{self.args.ckpt2test.replace('.pth', suffix)}")



if __name__ == "__main__":
    tranier = Trainer()
    tranier.train()
    # tranier.test(eval_on='uav',load_ckpt=True)
    tranier.test2loc(eval_on='uav',load_ckpt=True)
    # tranier.test_seq()







