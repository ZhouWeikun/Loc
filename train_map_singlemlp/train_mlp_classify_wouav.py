import torch
import argparse
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from mertic_learning import LpDistance

from get_img_encoder import ImgEncoder
from dataset_map_learner_nrc_center import SatMapDataset

def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")
    parser.add_argument('--p_satinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu__512h_lineClassed/dataset_xmu_meta/satimg_xiangan_xmu_info.json',
                        type=str, help='training dir path')
    parser.add_argument("--n_epoch", nargs="?", type=int, default=1000, help="Number of training steps")
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--num_worker', default=8, type=int, help='batchsize')
    parser.add_argument('--tensorboard', action='store_true', default = True)
    parser.add_argument('--exp_name', default='debug', type=str)
    parser.add_argument('--checkpoint', default="/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/epoch_5.pth", type=str, help='path for testing') # for testing
    parser.add_argument('--load_from', default="", type=str, help='exps path for pre-loading') #for continuing training
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
        self.dataset = SatMapDataset(
            p_satinfo_json='/home/data/zwk/data_uavimgs_XianganXmu__512h_lineClassed/dataset_xmu_meta/satimgs_xiangan_xmu_info.json',
            rand_rot_sat=False,
            )
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.args.batchsize, num_workers=self.args.num_worker,
                                                 pin_memory=True, shuffle=True, drop_last=True)
        self.grid_centers = self.dataset.mk_coord_grid(split_by='hw',hw=(8,10),random=False).reshape(-1,2).to(device)
        self.dist2gaussian = torch.distributions.Normal(loc=0, scale=self.dataset.grid_cell_radius*0.5) #2sigma that contains 99% pdf = grid_cell_radius
        self.eucdist_computer = LpDistance(normalize_embeddings=False)

        # load the trained img_encoder
        self.img_encoder = ImgEncoder()
        self.img_encoder._test_reay()

        # prepare the map_encoder
        # map_encoder_cfg = {
        # 	"otype": "CutlassMLP",#CutlassMLP or FullyFusedMLP
        # 	"activation": "ReLU",
        # 	"output_activation": "None",
        # 	"n_neurons": 512,
        # 	"n_hidden_layers": 4
        # }
        # map_encoder = tcnn.Network(n_input_dims=512+2, n_output_dims=1,network_config=map_encoder_cfg)
        from pos_encoder import PositionalEncoder
        pos_encoder = PositionalEncoder(multires=0, input_dims=2)
        self.pos_encoder = pos_encoder

        from mlp_creater import create_mlp, init_weights
        dims = [512 + pos_encoder.out_dim, 512, 512, 8 * 10]
        activation = nn.LeakyReLU
        map_encoder = create_mlp(dims, activation_fn=activation, norm_type='layer').to(device)
        map_encoder.apply(lambda m: init_weights(m, method='kaiming', nonlinearity='leaky_relu'))
        self.map_encoder = map_encoder

    def test(self):
        args = self.args
        device = self.device
        dataloader = self.dataloader
        pos_encoder = self.pos_encoder
        map_encoder = self.map_encoder
        img_encoder = self.img_encoder
        eucdist_computer = self.eucdist_computer
        grid_centers = self.grid_centers
        dist2gaussian = self.dist2gaussian

        checkpoint = torch.load(args.checkpoint, map_location=device)  # map_location确保能正确加载到CPU或GPU
        map_encoder.load_state_dict(checkpoint['model_state_dict'])
        map_encoder.eval()

        for it, data in tqdm(enumerate(dataloader)):
            nrcs, imgs = data
            feats = img_encoder.model(imgs.to(device))
            nrc_encoded = pos_encoder(nrcs)
            input = torch.concatenate([nrc_encoded.to(device), feats.expand(nrc_encoded.shape[0], -1)], -1)
            output = map_encoder(input)
            pred_pdf = torch.softmax(output, dim=-1)

            nrcs2centers = eucdist_computer(nrcs.to(device), grid_centers)
            gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers))
            gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)

            id_gt = torch.argmax(gt_pdf, dim=-1, keepdim=False)
            id_pred = torch.argmax(pred_pdf, dim=-1, keepdim=False)
            recall_1 = (id_pred==id_gt).sum()/id_pred.shape[0]
            print(f"recall={recall_1:.4f}")
            pred_rcs = grid_centers[id_pred]
            dist_rel2radius = torch.norm(pred_rcs-nrcs.to(device),dim=-1)/dataloader.dataset.grid_cell_radius
            dist_rel2radius_recall_1 = dist_rel2radius.mean()
            print(f"dist_rel2radius_recall_1={dist_rel2radius_recall_1:.4f}")
            exit()
            # debug:
            # from matplotlib import pyplot as plt
            # # 设定一个统一的数值范围，比如概率范围是0到1
            # pdf_min, pdf_max = 0, 1
            # diff_max = torch.abs(pred_pdf - gt_pdf).max().item()
            # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))  # figsize设置整个画布的大小
            # im1 = ax[0].imshow(pred_pdf.detach().cpu().numpy().squeeze(),cmap='viridis')
            # im2 = ax[1].imshow(gt_pdf.detach().cpu().numpy().squeeze(),cmap='viridis')
            # im3 = ax[2].imshow(torch.abs(pred_pdf - gt_pdf).detach().cpu().numpy().squeeze(), cmap='inferno')
            # cbar3 = fig.colorbar(im3, ax=ax[2])
            # plt.savefig("/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/predvsgt.png")
            # plt.show()

    def train(self):
        args = self.args
        device = self.device
        dataloader = self.dataloader
        pos_encoder = self.pos_encoder
        map_encoder = self.map_encoder
        img_encoder = self.img_encoder
        eucdist_computer = self.eucdist_computer
        grid_centers = self.grid_centers
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

        writer = SummaryWriter(
            f"/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/{args.exp_name}/train_tensorboard.log") if args.tensorboard else None

        for epoch in range(args.n_epoch):
            for it, data in tqdm(enumerate(dataloader)):
                nrcs, imgs = data
                feats = img_encoder.model(imgs.to(device))

                nrc_encoded = pos_encoder(nrcs)
                input = torch.concatenate([nrc_encoded.to(device), feats.expand(nrc_encoded.shape[0], -1)], -1)
                output = map_encoder(input)
                pred_pdf = torch.softmax(output,dim=-1)

                nrcs2centers = eucdist_computer(nrcs.to(device),grid_centers)
                gt_pdf = torch.exp(dist2gaussian.log_prob(nrcs2centers))
                gt_pdf = gt_pdf / gt_pdf.sum(dim=-1, keepdim=True)
                loss = -gt_pdf*torch.log(pred_pdf)
                loss = loss.mean()

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

            if epoch % 5 == 0:
                dir2save = "/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': map_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,  # 可选：保存最后一个 batch 的损失值
                }, os.path.join(dir2save, f"epoch_{epoch}.pth"))




if __name__ == "__main__":
    torch.manual_seed(666)
    tranier = Trainer()
    # tranier.train()
    tranier.test()







