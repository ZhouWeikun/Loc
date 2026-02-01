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
from dataset_map_learner_xmu import SatMapDataset, UAVDataset
# from dataset_map_learner_gta import SatMapDataset, UAVDataset
torch.manual_seed(2025)
np.random.seed(2025)

from vis_featmap import vis_multi_featmap,vis_single_featmap
from eval_recall_fm_salad import compute_recall_by_label
from loc_utils import agg_seq_pdf,find_4neighbors_topleft,compute_agged_pred_4neighbors_id


def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")
    parser.add_argument('--exp_name', default='debug_xmu_mcdropout', type=str)
    parser.add_argument('--p_satinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu_512h_lineClassed/dataset_xmu_meta/satimg_xiangan_xmu5km2_03res_info.json',
                        type=str, help='training sat dir path')
    parser.add_argument('--p_uavinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu_512h_lineClassed/dataset_xmu_meta/uavimgs_xiangan_xmu_info.json',
                        type=str, help='training uav dir path')
    parser.add_argument('--p_img_enccoder_cfg',
                        default='/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug_xmu_vits_224/opts.yaml',
                        type=str, help='training uav dir path')
    parser.add_argument('--p_img_encoder_ckpt',
                        default='/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug_xmu_vits_224/epoch002.pth',
                        type=str, help='training uav dir path')
    parser.add_argument('--satimgsize2clip',default = 224, type=int, help='the satimg cliped from tif')
    parser.add_argument('--batchsize_sat', default=1024+512, type=int, help='batchsize')
    parser.add_argument('--batchsize_uav', default=256, type=int, help='batchsize')
    parser.add_argument("--n_epoch", nargs="?", type=int, default=100, help="Number of training steps")
    parser.add_argument('--num_worker', default=8, type=int, help='batchsize')
    parser.add_argument('--tensorboard', action='store_true', default = False)
    parser.add_argument('--ckpt2test', default="/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_xmu_mcdropout/epoch_29.pth", type=str, help='path for testing') # for testing
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
            satimgsize2clip=self.args.satimgsize2clip,
            rand_rot_sat=True,
            stage='train',
        )
        self.sat_dataloader = torch.utils.data.DataLoader(self.sat_dataset, batch_size=self.args.batchsize_sat,
                                                          num_workers=self.args.num_worker,
                                                          pin_memory=True, shuffle=True, drop_last=True,
                                                          persistent_workers=True)
        self.uav_dataset = UAVDataset(
            p_uavinfo_json=self.args.p_uavinfo_json,
            stage='train',
        )
        self.uav_dataset.mk_nrcs_fm_latlons(satmap_dataset=self.sat_dataset)
        self.uav_dataloader_train = torch.utils.data.DataLoader(self.uav_dataset, batch_size=self.args.batchsize_uav,
                                                                num_workers=self.args.num_worker,
                                                                pin_memory=True, shuffle=True, drop_last=False,
                                                                persistent_workers=True)
        self.uav_dataloader_test = torch.utils.data.DataLoader(self.uav_dataset, batch_size=self.args.batchsize_uav,
                                                               num_workers=self.args.num_worker,
                                                               pin_memory=True, shuffle=False, drop_last=False,
                                                               persistent_workers=True)

        self.grid_centers = self.sat_dataset.mk_coord_grid(split_by='hw', hw=(8, 10), random=False).reshape(-1, 2).to(device)
        self.grid_n_hw = (self.sat_dataset.n_grid_h,self.sat_dataset.n_grid_w)
        self.dist2gaussian = torch.distributions.Normal(loc=0,
                                                        scale=self.sat_dataset.grid_cell_radius * 0.65)  # scale=sigma, 2sigma that contains 99% pdf = grid_cell_radius
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
        self.img_encoder = ImgEncoder(p2yaml='/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug_xmu_vits_224/opts.yaml')
        self.img_encoder._test_reay(self.args.p_img_enccoder_cfg,self.args.p_img_encoder_ckpt)

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
        input_dim = 384
        self.map_encoder = MultiMLP(self.grid_centers.shape[0],input_dim=input_dim,mlp_hidden_dims=[ 128, 32, 8],
                            mlp_activation_fn=nn.LeakyReLU,mlp_norm_type='layer', mlp_dropout_p=0.25,
                            mlp_init_method='kaiming',mlp_init_nonlinearity='leaky_relu',
                            device=self.device)
        self.map_encoder.p_dropout=0.25


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
        grid_centers = self.grid_centers
        dist2gaussian = self.dist2gaussian

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

        #config for recording exp
        expdir2save = "exps/{}".format(self.args.exp_name)
        if not os.path.exists(expdir2save):
            os.mkdir(expdir2save)
        self.writer = SummaryWriter(f"{expdir2save}/{args.exp_name}/train_tensorboard.log") if args.tensorboard else None
        from tool.utils_fm_duav import  get_logger
        self.logger = get_logger("exps/{}/train.log".format(self.args.exp_name),'trainer_logger')

        for epoch in torch.range(epoch_begin,args.n_epoch,dtype=torch.uint8):
            for it, data in tqdm(enumerate(self.sat_dataloader)):
                nrcs, imgs = data
                uav_nrcs, uav_imgs = next(iter(self.uav_dataloader_train))
                nrcs = torch.concatenate([nrcs,uav_nrcs],dim=0).to(device)
                imgs = torch.concatenate([imgs,uav_imgs],dim=0).to(device)

                feats = img_encoder.model(imgs)
                # nrc_encoded = pos_encoder(nrcs)
                # input = torch.concatenate([nrc_encoded, feats.expand(nrc_encoded.shape[0], -1)], -1)
                input = feats
                output = map_encoder(input)
                pred_pdf = torch.softmax(output,dim = -1)

                nrcs2centers = eucdist_computer_rc(nrcs,grid_centers)
                gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers))
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
            self.map_encoder.train()
            self.img_encoder.model.train()
            self.uav_dataloader_train.dataset.switch_stage('train')
            self.writer.add_scalar('recall1', recall1, epoch) if self.writer is not None else None
            self.writer.add_scalar('recall1_rel_dist', recall1_rel_dist, epoch) if self.writer is not None else None
            # print(f"ep={epoch}; recall1={recall1:.4f}; recall1_rel_dist={recall1_rel_dist:.4f}")
            self.logger.info(f"ep={epoch} finished; recall1={recall1:.4f}; recall1_rel_dist={recall1_rel_dist:.4f}") if hasattr(self,'logger') else None
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
            grid_centers = self.grid_centers
            dist2gaussian = self.dist2gaussian

            if load_ckpt:
                checkpoint = torch.load(args.ckpt2test, map_location=device)  # map_location确保能正确加载到CPU或GPU
                map_encoder.load_state_dict(checkpoint['model_state_dict'])
            img_encoder.model.eval()
            map_encoder.train() if map_encoder.p_dropout>0 else map_encoder.model.eval()

            #test on uav_imgs
            if eval_on == 'uav':
                dataloader = uav_dataloader
            else:
                dataloader = sat_dataloader
            dataloader.dataset.switch_stage('test')

            recall1_list,dist_recall1_list = [],[]
            q_label_list, pred_classid_list = [],[]
            id_4neighbors_list = []
            pred_pdf_list = []
            for it, data in tqdm(enumerate(dataloader)):
                nrcs, imgs = data
                nrcs, imgs = nrcs.to(device), imgs.to(device)

                feats = img_encoder.model(imgs)
                # nrc_encoded = pos_encoder(nrcs)
                # input = torch.concatenate([nrc_encoded, feats.expand(nrc_encoded.shape[0], -1)], -1)
                input = feats
                if self.map_encoder.p_dropout>0:
                    batch_predictions = []
                    for _ in range(50):
                        output = map_encoder(input)
                        pred_pdf = torch.softmax(output, dim=-1)
                        batch_predictions.append(pred_pdf)
                    stacked_preds = torch.stack(batch_predictions, dim=1)
                    mean_prediction = torch.mean(stacked_preds, dim=1)
                    std_prediction = torch.std(stacked_preds, dim=1)
                    pred_pdf = std_prediction
                    #debug for vis
                    # vis_single_featmap(mean_prediction[0].reshape(self.grid_n_hw[0],self.grid_n_hw[1]),'/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/dropout_mean.png')
                    # vis_single_featmap(std_prediction[0].reshape(self.grid_n_hw[0],self.grid_n_hw[1]),'/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/dropout_mean.png')

                # debug 4 recall@k testing for single loc:
                nrcs2centers = eucdist_computer_rc(nrcs, grid_centers)
                # nrcs2centers_np = nrcs2centers.cpu().numpy()
                gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers))
                gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)

                #debug:
                # vis_multi_featmap(mean_prediction[:9].reshape(-1,self.grid_n_hw[0],self.grid_n_hw[1]),'/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/dropout_mean.png')
                # vis_multi_featmap(std_prediction[:9].reshape(-1,self.grid_n_hw[0],self.grid_n_hw[1]),'/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/dropout_std.png')

                # vis_multi_featmap(nrcs2centers.reshape(-1, 8, 10)[:9], p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/debug_rcdist.png')
                # vis_multi_featmap(gt_pdf[:9].reshape(-1,self.grid_n_hw[0],self.grid_n_hw[1]), p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_map_mutimlp/exps/debug_vis/debug_gtpdf.png',interpolation='bilinear')
                # vis_multi_featmap(pred_pdf.reshape(-1, 8, 10)[:9],
                #                   p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/debug_predpdf.png')

                recall1, dist_recall1 = self.compute_recall(pred_pdf, gt_pdf, nrcs)
                recall1_list.append(recall1)
                dist_recall1_list.append(dist_recall1)
                q_labels = torch.argmax(gt_pdf, dim=-1)
                q_label_list.append(q_labels)
                orted_values, sorted_indices = torch.sort(pred_pdf, descending=True)
                pred_classid_list.append(sorted_indices)
                pred_pdf_list.append(pred_pdf)

                # debug 4 recall@k testing for single loc:
                # from eval_recall_fm_salad import compute_recall_by_label
                # q_labels = torch.argmax(gt_pdf, dim=-1)
                # pred_vals_per_query,pred_labels_per_query = torch.sort(pred_pdf, dim=-1,descending=True)
                # pred_dict = compute_recall_by_label(q_labels.detach().cpu().numpy(),pred_labels_per_query.detach().cpu().numpy(),[1,2,3,4,5,10])

                id_4neighbors_flat = compute_agged_pred_4neighbors_id(pred_pdf.reshape(-1,8,10))
                id_4neighbors_list.append(id_4neighbors_flat)

                # from vis_featmap import vis_multi_featmap
                # featmap = pred_pdf[:16].reshape(16,8,10)
                # vis_multi_featmap(featmap,p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/pred_pdf.png',interpolation='nearest')
                # featmap = gt_pdf[9:16+9].reshape(16,8,10)
                # vis_multi_featmap(featmap,p2save='/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/vis/gt_pdf.png',interpolation='nearest')


            recall1 = torch.tensor(recall1_list).mean()
            recall1_rel_dist = torch.tensor(dist_recall1_list).mean()
            print(f"recall1={recall1:.4f}")
            print(f"dist_recall1={recall1_rel_dist:.4f}")

            q_label_list = torch.cat(q_label_list).cpu().numpy()
            pred_classid_list = torch.cat(pred_classid_list).cpu().numpy()
            single_loc_recall = compute_recall_by_label(q_label_list,pred_classid_list,[1,2,3,4,5],title='single_loc_recall')
            id_4neighbors_list = torch.cat(id_4neighbors_list).cpu().numpy()
            single_loc_4neighbor_recall = compute_recall_by_label(q_label_list, id_4neighbors_list, [1, 2, 3, 4],title='single_loc_4neighbor_recall')
            pred_pdf_list = torch.cat(pred_pdf_list)
            pred_seq_agged = agg_seq_pdf(pred_pdf_list)
            id_4neighbors_flat = compute_agged_pred_4neighbors_id(pred_seq_agged.reshape(-1,8,10)).cpu().numpy()
            seq_agged_loc_4neighbor_recall = compute_recall_by_label(q_label_list[-id_4neighbors_flat.shape[0]:], id_4neighbors_flat, [1, 2, 3, 4],title='seq_agged_loc_4neighbor_recall')

            if hasattr(self,'logger'):
                info2log = {"single_loc_recall":single_loc_recall,"single_loc_4neighbor_recall":single_loc_4neighbor_recall,"seq_agged_loc_4neighbor_recall":seq_agged_loc_4neighbor_recall}
                for key in info2log.keys():
                    info = key
                    for k, v in info2log[key].items():
                        info = info + f" @{k}:{v * 100:.2f} "
                    self.logger.info(info)

            return recall1,recall1_rel_dist


    def compute_recall(self, pred_pdf, gt_pdf, gt_nrcs):
        id_gt = torch.argmax(gt_pdf, dim=-1, keepdim=False)
        id_pred = torch.argmax(pred_pdf, dim=-1, keepdim=False)
        recall_1 = (id_pred == id_gt).sum() / id_pred.shape[0]
        # print(f"recall={recall_1:.4f}")
        pred_rcs = self.grid_centers[id_pred]
        dist_rel2radius = torch.norm(pred_rcs - gt_nrcs, dim=-1) / self.sat_dataloader.dataset.grid_cell_radius
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
    # tranier.train()
    tranier.test(eval_on='uav',load_ckpt=True)
    # tranier.test_seq()







