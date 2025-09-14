import numpy as np
import torch
import argparse
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from mertic_learning import LpDistance
import torch.nn.functional as F
import numpy as np

from get_img_encoder import ImgEncoder
# from dataset_map_learner_xmu import SatMapDataset, UAVDataset
from dataset_map_learner_gta import SatMapDataset, UAVDataset
torch.manual_seed(2025)
np.random.seed(2025)

from vis_featmap import vis_multi_featmap,vis_single_featmap
from eval_recall_fm_salad import compute_recall_by_label
from loc_utils import agg_seq_pdf,find_4neighbors_topleft,compute_agged_pred_4neighbors_id,find_nneighbors_topleft,compute_agged_pred_nneighbors_id


def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")
    parser.add_argument('--exp_name', default='debug_gta', type=str)
    parser.add_argument('--p_satinfo_json',
                        default='/home/data/zwk/dataset_Game4Loc/satmap_info.json',
                        type=str, help='training sat dir path')
    parser.add_argument('--p_uavinfo_json',
                        default='/home/data/zwk/dataset_Game4Loc/uavimg_info.json',
                        type=str, help='training uav dir path')
    parser.add_argument('--p_img_enccoder_cfg',
                        default='/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug10_gta_vitb_224/opts.yaml',
                        type=str, help='training uav dir path')
    parser.add_argument('--p_img_encoder_ckpt',
                        default='/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug10_gta_vitb_224/epoch000.pth',
                        type=str, help='training uav dir path')
    parser.add_argument('--imgsize2net',default=224, type=int, help='the satimg cliped from tif')
    parser.add_argument('--batchsize_sat', default=1024+512, type=int, help='batchsize')
    parser.add_argument('--batchsize_uav', default=128, type=int, help='batchsize')
    parser.add_argument("--n_epoch", nargs="?", type=int, default=100, help="Number of training steps")
    parser.add_argument('--num_worker', default=4, type=int, help='batchsize')
    parser.add_argument('--tensorboard', action='store_true', default = False)
    parser.add_argument('--ckpt2test', default="/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_gta/epoch_5.pth", type=str, help='path for testing') # for testing
    parser.add_argument('--ckpt2train', default="", type=str, help='exps path for pre-loading') #for continuing training
    return parser.parse_args()


class Trainer(object):
    def __init__(self,):
        self.args = get_args()
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            self.args.use_gpu = True
        else:
            device = torch.device("cpu")
            self.args.use_gpu = False
        self.device = device

        # prepare the dataset
        self.sat_dataset = SatMapDataset(
            p_satinfo_json=self.args.p_satinfo_json,
            # imgsize2clip=self.args.satimgsize2clip,
            # rand_rot_sat=True,
            imgsize2net=self.args.imgsize2net,
            stage='train',
            )
        self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset, batch_size=self.args.batchsize_sat, num_workers=self.args.num_worker,
                                                 pin_memory=True, shuffle=True, drop_last=True)
        self.uav_dataset_train = UAVDataset(
            p_uavinfo_json=self.args.p_uavinfo_json,
            p_pairs_json='/home/data/zwk/dataset_Game4Loc/cross-area-drone2sate-train.json',
            imgsize2net=self.args.imgsize2net,
            stage='train',
        )
        # self.uav_dataset.mk_nrcs_fm_latlons(satmap_dataset=self.sat_dataset)
        self.uav_dataloader_train = torch.utils.data.DataLoader(self.uav_dataset_train, batch_size=self.args.batchsize_uav, num_workers=self.args.num_worker,
                                                 pin_memory=True, shuffle=True, drop_last=False)
        self.uav_dataset_test = UAVDataset(
            p_uavinfo_json=self.args.p_uavinfo_json,
            p_pairs_json='/home/data/zwk/dataset_Game4Loc/cross-area-drone2sate-test.json',
            imgsize2net=self.args.imgsize2net,
            stage='test',
        )
        self.uav_dataloader_test = torch.utils.data.DataLoader(self.uav_dataset_test, batch_size=self.args.batchsize_uav, num_workers=self.args.num_worker,
                                                 pin_memory=True, shuffle=False, drop_last=False)

        self.grid_centers_4sample = self.sat_dataset.grid_centers2sample.to(self.device)
        self.grid_centers_all = self.sat_dataset.grid_centers.to(self.device)
        self.grid_cell_radius = self.sat_dataset.grid_cell_radius
        self.grid_pdf_all = torch.zeros(self.sat_dataset.grid_centers.shape[:2]).to(self.device)
        self.grid_mask = self.sat_dataset.grid_mask
        self.dist2gaussian = torch.distributions.Normal(loc=0, scale=0.5*(self.sat_dataset.grid_cell_hw[0]+self.sat_dataset.grid_cell_hw[1])*0.65) #scale=sigma, 2sigma that contains 99% pdf = grid_cell_radius
        self.eucdist_computer_feat = LpDistance(normalize_embeddings=True)
        self.eucdist_computer_rc = LpDistance(normalize_embeddings=False)

        # debug,vis the gird
        # from vis_loc_res import vis_rcs_on_tif
        # pts = torch.cat(
        #     [self.grid_centers.detach().cpu(), self.sat_dataset.nrc_boundary_meshgrid.reshape(-1, 2)]).numpy()
        # vis_rcs_on_tif(np.array(self.sat_dataset.tif_img), self.grid_centers.detach().cpu().numpy(),
        #                p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/grid_centers.png',
        #                color='red')
        # vis_rcs_on_tif(np.array(self.sat_dataset.tif_img),
        #                self.sat_dataset.nrc_boundary_meshgrid.reshape(-1, 2).detach().cpu().numpy(),
        #                p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/nrc_boundary_meshgrid.png',
        #                color='blue')
        # vis_rcs_on_tif(np.array(self.sat_dataset.tif_img), pts,
        #                p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/grid_cneter&boundary',
        #                color='blue')

        # load the trained img_encoder, version1:
        self.img_encoder = ImgEncoder()
        self.img_encoder._test_reay(self.args.p_img_enccoder_cfg,self.args.p_img_encoder_ckpt)
        # load the game4loc model, version2:
        # from load_gta_model import GTAVIT
        # self.img_encoder = GTAVIT(device=self.device)

        # prepare the map_encoder
        #   version1,by tinndy_cuda_nn
        # map_encoder_cfg = {
        # 	"otype": "CutlassMLP",#CutlassMLP or FullyFusedMLP
        # 	"activation": "ReLU",
        # 	"output_activation": "None",
        # 	"n_neurons": 512,
        # 	"n_hidden_layers": 4
        # }
        # map_encoder = tcnn.Network(n_input_dims=512+2, n_output_dims=1,network_config=map_encoder_cfg)
        #   version2: by single mlp:
        from get_pos_encoder import PositionalEncoder
        # pos_encoder = PositionalEncoder(multires=0, input_dims=2)
        # self.pos_encoder = pos_encoder
        # from mlp_creater import create_mlp, init_weights
        # input_dim = self.img_encoder.get_output_dim() + pos_encoder.out_dim
        # dims = [input_dim, 128, 32, 8, 1] #when vit't output_dim = 384
        # map_encoder = create_mlp(dims, activation_fn=nn.LeakyReLU, norm_type='layer').to(device)
        # map_encoder.apply(lambda m: init_weights(m, method='kaiming', nonlinearity='leaky_relu'))
        # self.map_encoder = map_encoder
        #   version3: by multi-mlp:
        from get_pos_encoder import PositionalEncoder
        pos_encoder = PositionalEncoder(multires=0, input_dims=2)
        self.pos_encoder = pos_encoder
        from get_mlps_classify import create_mlp, init_weights, MultiMLP
        # input_dim = self.img_encoder.get_output_dim() + pos_encoder.out_dim
        input_dim = 768
        # mlp_hidden_dims = [256,128,64]
        mlp_hidden_dims = [128,64,32]

        self.map_encoder = MultiMLP(self.grid_centers_4sample.shape[0],input_dim=input_dim,mlp_hidden_dims=mlp_hidden_dims,
                            mlp_activation_fn=nn.LeakyReLU,mlp_norm_type='layer', mlp_dropout_p=self.map_encoder.p_dropout,
                            mlp_init_method='kaiming',mlp_init_nonlinearity='leaky_relu',
                            device=self.device)


    def train(self):
        args = self.args
        device = self.device
        # sat_dataloader = self.sat_dataloader
        # uav_dataloader = self.uav_dataloader_train
        pos_encoder = self.pos_encoder
        map_encoder = self.map_encoder
        img_encoder = self.img_encoder
        # eucdist_computer_feat = self.eucdist_computer_feat
        eucdist_computer_rc = self.eucdist_computer_rc
        grid_centers_4sample = self.grid_centers_4sample
        dist2gaussian = self.dist2gaussian

        optimizer_cfg = {
            "otype": "Adam",
            "lr": 1e-2,
            "beta1": 0.9,
            "beta2": 0.99,
            "eps": 1e-8,
            "l2_reg": 1e-8
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

        #config for recording exp
        expdir2save = "exps/{}".format(self.args.exp_name)
        if not os.path.exists(expdir2save):
            os.mkdir(expdir2save)
        self.writer = SummaryWriter(f"{expdir2save}/{args.exp_name}/train_tensorboard.log") if args.tensorboard else None
        from tool.utils import  get_logger
        self.logger = get_logger("exps/{}/train.log".format(self.args.exp_name),'trainer_logger')

        for epoch in torch.range(epoch_begin,args.n_epoch,dtype=torch.uint8):
            for it, data in tqdm(enumerate(self.sat_dataloader)):
                nrcs, imgs = data

                uav_nrcs, uav_imgs = next(iter(self.uav_dataloader_train))
                nrcs = torch.concatenate([nrcs,uav_nrcs],dim=0).to(device)
                imgs = torch.concatenate([imgs,uav_imgs],dim=0).to(device)

                feats = img_encoder.model(imgs) if hasattr(img_encoder, 'model') else img_encoder(imgs)
                # nrc_encoded = pos_encoder(nrcs) if pos_encoder.kwargs['num_freqs']>0 else nrcs
                # input = torch.concatenate([nrc_encoded, feats.expand(nrc_encoded.shape[0], -1)], -1)
                input = feats
                output = map_encoder(input)
                pred_pdf = torch.softmax(output,dim = -1)

                nrcs2centers_4sample = eucdist_computer_rc(nrcs,grid_centers_4sample)
                gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers_4sample))
                gt_pdf = gt_pdf / gt_pdf.sum(dim = -1, keepdim = True)
                loss = - gt_pdf * torch.log(pred_pdf)
                loss = loss.sum(dim=-1).mean()
                # loss = loss.mean()
                # grid_centers_np = grid_centers.cpu().numpy()
                # nrcs2centers_0 = nrcs2centers.cpu().numpy()[0].reshape(8,10)
                # gt_pdf_0 = gt_pdf.cpu().numpy()[0].reshape(8,10)

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

            recall1,recall1_rel_dist = self.test(eval_on='uav',load_ckpt=False)
            self.logger.info(f"ep={epoch} finished; recall1={recall1:.4f}; recall1_rel_dist={recall1_rel_dist:.4f}")
            self.map_encoder.train()
            self.img_encoder.model.train()
            self.uav_dataloader_train.dataset.switch_stage('train')
            self.writer.add_scalar('recall1', recall1, epoch) if self.writer is not None else None
            self.writer.add_scalar('recall1_rel_dist', recall1_rel_dist, epoch) if self.writer is not None else None
            # print(f"ep={epoch}; recall1={recall1:.4f}; recall1_rel_dist={recall1_rel_dist:.4f}")
            # map_encoder.train()
            # img_encoder.model.train()

            # if epoch % 5 == 0:
            dir2save = f"/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/{args.exp_name}"
            torch.save({
                'epoch': epoch,
                'model_state_dict': map_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,  # 可选：保存最后一个 batch 的损失值
            }, os.path.join(dir2save, f"epoch_{epoch}.pth"))


    def test(self,eval_on='uav',load_ckpt=False):
        with torch.no_grad():
            args = self.args
            device = self.device
            sat_dataloader = self.sat_dataloader
            uav_dataloader = self.uav_dataloader_test
            pos_encoder = self.pos_encoder
            map_encoder = self.map_encoder
            img_encoder = self.img_encoder
            eucdist_computer_feat = self.eucdist_computer_feat
            eucdist_computer_rc = self.eucdist_computer_rc
            grid_centers_4sample = self.grid_centers_4sample
            dist2gaussian = self.dist2gaussian

            if load_ckpt:
                checkpoint = torch.load(args.ckpt2test, map_location=device)  # map_location确保能正确加载到CPU或GPU
                map_encoder.load_state_dict(checkpoint['model_state_dict'])
            map_encoder.eval()
            img_encoder.model.eval()

            #test on uav_imgs
            if eval_on == 'uav':
                dataloader = uav_dataloader
            else:
                dataloader = sat_dataloader
            dataloader.dataset.switch_stage('test')

            recall1_list,dist_recall1_list = [],[]
            q_label_list, pred_classid_list = [],[]
            q_label_all_list = []
            id_4neighbors_list = []
            pred_pdf_list = []
            pred_pdf_all_list = []
            pred_classid_all_list = []
            for it, data in tqdm(enumerate(dataloader)):
                nrcs, imgs = data
                nrcs, imgs = nrcs.to(device), imgs.to(device)

                #predict the res
                feats = img_encoder.model(imgs)
                # nrc_encoded = pos_encoder(nrcs)
                # input = torch.concatenate([nrc_encoded, feats.expand(nrc_encoded.shape[0], -1)], -1)
                input = feats
                output = map_encoder(input)
                pred_pdf = torch.softmax(output, dim=-1)

                nrcs2centers_4sample = eucdist_computer_rc(nrcs, grid_centers_4sample)
                gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers_4sample))
                gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)

                #for recall1_rel_dist
                recall1, dist_recall1 = self.compute_recall(pred_pdf, gt_pdf, nrcs,self.grid_centers_4sample,self.grid_cell_radius)
                recall1_list.append(recall1)
                dist_recall1_list.append(dist_recall1)

                #for single_loc_recall res
                q_labels = torch.argmax(gt_pdf, dim=-1)
                q_label_list.append(q_labels)
                orted_values, sorted_indices = torch.sort(pred_pdf, descending=True)
                pred_classid_list.append(sorted_indices)
                pred_pdf_list.append(pred_pdf)
                #for single_loc_4neighbor_recall res
                pred_pdf_all = self.grid_pdf_all.unsqueeze(0).repeat(pred_pdf.shape[0], 1,1)
                pred_pdf_all[:,self.grid_mask] = pred_pdf
                id_4neighbors_flat = compute_agged_pred_nneighbors_id(pred_pdf_all,n=3)
                id_4neighbors_list.append(id_4neighbors_flat)
                nrcs2centers_all = eucdist_computer_rc(nrcs, self.grid_centers_all.reshape(-1,2))
                gt_pdf_all = torch.exp(dist2gaussian.log_prob(nrcs2centers_all))
                gt_pdf_all = gt_pdf_all / gt_pdf_all.sum(dim=-1, keepdim=True)
                q_labels_all = torch.argmax(gt_pdf_all, dim=-1)
                q_label_all_list.append(q_labels_all)
                #for seq_agged_loc_4neighbor_recall res
                # orted_values, sorted_indices = torch.sort(pred_pdf_all, descending=True)
                # pred_classid_all_list.append(sorted_indices)
                # pred_pdf_all_list.append(pred_pdf_all)


            recall1 = torch.tensor(recall1_list).mean()
            recall1_rel_dist = torch.tensor(dist_recall1_list).mean()
            print(f"recall1={recall1:.4f}")
            print(f"dist_recall1={recall1_rel_dist:.4f}")

            q_label_list = torch.cat(q_label_list).cpu().numpy()
            pred_classid_list = torch.cat(pred_classid_list).cpu().numpy()
            single_loc_recall = compute_recall_by_label(q_label_list,pred_classid_list,[1,2,3,4,5],title='single_loc_recall')

            q_label_all_list = torch.cat(q_label_all_list).cpu().numpy()
            id_4neighbors_list = torch.cat(id_4neighbors_list).cpu().numpy()
            single_loc_4neighbor_recall = compute_recall_by_label(q_label_all_list, id_4neighbors_list, [1, 2, 3, 4, 9],title='single_loc_neighbor_recall')

            # pred_pdf_all_list = torch.cat(pred_pdf_all_list)
            # pred_all_seq_agged = agg_seq_pdf(pred_pdf_all_list.reshape(pred_pdf_all_list.shape[0], -1),window_len=3)
            # id_4neighbors_flat = compute_agged_pred_4neighbors_id(pred_all_seq_agged.reshape(-1,*pred_pdf_all_list.shape[-2:])).cpu().numpy()
            # seq_agged_loc_4neighbor_recall = compute_recall_by_label(q_label_all_list[-id_4neighbors_flat.shape[0]:], id_4neighbors_flat, [1, 2, 3, 4],title='seq_agged_loc_4neighbor_recall')

            if hasattr(self,'logger'):
                # info2log = {"single_loc_recall":single_loc_recall,"single_loc_4neighbor_recall":single_loc_4neighbor_recall,"seq_agged_loc_4neighbor_recall":seq_agged_loc_4neighbor_recall}
                info2log = {"single_loc_recall": single_loc_recall,"single_loc_4neighbor_recall":single_loc_4neighbor_recall}

                for key in info2log.keys():
                    info = key
                    for k, v in info2log[key].items():
                        info = info + f" @{k}:{v * 100:.2f} "
                    self.logger.info(info)

            return recall1,recall1_rel_dist


    def compute_recall(self, pred_pdf, gt_pdf, gt_nrcs,grid_centers,grid_cell_radius):
        id_gt = torch.argmax(gt_pdf, dim=-1, keepdim=False)
        id_pred = torch.argmax(pred_pdf, dim=-1, keepdim=False)
        recall_1 = (id_pred == id_gt).sum() / id_pred.shape[0]
        # print(f"recall={recall_1:.4f}")
        pred_rcs = grid_centers[id_pred]
        dist_rel2radius = torch.norm(pred_rcs - gt_nrcs, dim=-1) / grid_cell_radius
        dist_rel2radius_recall_1 = dist_rel2radius.mean()
        # print(f"dist_rel2radius_recall_1={dist_rel2radius_recall_1:.4f}")
        return recall_1,dist_rel2radius_recall_1


    def test_seq(self):
        with torch.no_grad():
            args = self.args
            device = self.device
            sat_dataloader = self.sat_dataloader
            uav_dataloader = self.uav_dataloader
            pos_encoder = self.pos_encoder
            map_encoder = self.map_encoder
            img_encoder = self.img_encoder
            eucdist_computer_feat = self.eucdist_computer_feat
            eucdist_computer_rc = self.eucdist_computer_rc
            grid_centers = self.grid_centers
            dist2gaussian = self.dist2gaussian

            id_4neighbors_2recall = []
            gt_labels = []
            checkpoint = torch.load(args.ckpt2test, map_location=device)  # map_location确保能正确加载到CPU或GPU
            map_encoder.load_state_dict(checkpoint['model_state_dict'])
            dataloader = uav_dataloader
            dataloader.dataset.switch_stage('test')
            for it, data in tqdm(enumerate(dataloader)):
                nrcs, imgs = data
                nrcs, imgs = nrcs.to(device), imgs.to(device)

                feats_q = img_encoder.model(imgs)
                nrc_encoded = pos_encoder(nrcs)
                input = torch.concatenate([nrc_encoded, feats_q.expand(nrc_encoded.shape[0], -1)], -1)
                output = map_encoder(input)
                pred_pdf = torch.softmax(output, dim=-1)

                from loc_utils import agg_seq_pdf,find_4neighbors_topleft
                # pred_pdf_agged = agg_seq_pdf(pred_pdf)
                pred_pdf_agged = pred_pdf
                # debug
                # vis_single_featmap(pred_pdf_agged[0].reshape(8,10),p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/pred_pdf[0].png')

                # find the 4neighbors from the pred_pdf
                # id_topleft = find_4neighbors_topleft(pred_pdf_agged[0].reshape(8,10))

                #debug, compute the recal of 4neighbors
                id_toplefts = find_4neighbors_topleft(pred_pdf_agged.reshape(-1,8,10)).cpu()
                id_toprights = id_toplefts + torch.tensor([0,1])
                id_buttonlefts = id_toplefts + torch.tensor([1, 0])
                id_buttonrights = id_toplefts + torch.tensor([1, 1])
                id_4neighbors = torch.stack([id_toplefts,id_toprights,id_buttonlefts,id_buttonrights]).permute(1,0,2)
                id_4neighbors_flat = id_4neighbors[...,0]*10 + id_4neighbors[...,1]
                id_4neighbors_2recall.append(id_4neighbors_flat)

                nrcs2centers = eucdist_computer_rc(nrcs, grid_centers)
                gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers))
                gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)
                q_labels = torch.argmax(gt_pdf, dim=-1)
                gt_labels.append(q_labels.detach().cpu())

                # crop the sat_tif to sat_tiles
                # nrc_topleft = sat_dataloader.dataset.nrc_boundary_meshgrid[id_topleft[0],id_topleft[1]]
                # nrc_buttonright = sat_dataloader.dataset.nrc_boundary_meshgrid[id_topleft[0]+2,id_topleft[1]+2]
                # n2sample_h,n2sample_w = 128,128
                # sat_tiles,nrc_samples = sat_dataloader.dataset.sample_sats_in_rect(nrc_topleft,nrc_buttonright,n2sample_h=n2sample_h,n2sample_w=n2sample_w,satimgsize2clip=224,type2clip='tensor')

                # get the feature of  sat_tiles
                # split_size = 32
                # feat_gallery = []
                # for tiles in tqdm(torch.split(sat_tiles.reshape(-1,*sat_tiles.shape[2:]), split_size, dim=0)):
                #     output = img_encoder.model(tiles.to(device))
                #     feat_gallery.append(output.detach())
                # feat_gallery = torch.cat(feat_gallery,dim=0)  # .reshape(*rcs_girdcoord_center.shape[:2],-1)  # 形状 [N1*N2, D]

                # compute response map
                # feats_q = torch.nn.functional.normalize(feats_q,dim=-1)
                # feat_gallery = torch.nn.functional.normalize(feat_gallery,dim=-1)
                # dist = eucdist_computer_feat(feats_q[:9],feat_gallery)
                # dist_pdf = torch.exp(-dist)
                # gt_nrcs_in_grid = nrcs[:9].cpu()-nrc_topleft
                # gt_rowcols_in_grid = gt_nrcs_in_grid/(nrc_buttonright-nrc_topleft)*n2sample_h
                # id_max =torch.argmax(dist_pdf, dim=-1)
                # h_max,w_max = id_max // n2sample_h, id_max % n2sample_w
                # pred_position = torch.stack([h_max,w_max]).T
                # error = pred_position.cpu()-gt_rowcols_in_grid

                #debug
                # fig, axes = vis_multi_featmap(dist_pdf.reshape(-1,n2sample_h,n2sample_w),p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/dist_pdf_128hw.png')
                # from vis_featmap import add_gt2map
                # add_gt2map(axes,gt_rowcols_in_grid,'/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/dist&gt_pdf_128hw.png')
            id_4neighbors_2recall = torch.cat(id_4neighbors_2recall)
            gt_labels = torch.cat(gt_labels)
            pred_dict = compute_recall_by_label(gt_labels.numpy(),id_4neighbors_2recall.numpy(),[1,2,3,4])


if __name__ == "__main__":
    tranier = Trainer()
    tranier.train()
    # tranier.test(eval_on='uav',load_ckpt=True)
    # tranier.test_seq()







