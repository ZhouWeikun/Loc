import torch
import commentjson as json
import argparse
import torch.nn as nn

from torchvision import transforms

from train_img_encoder.trainer_retrival_relrot import get_parse as get_parse_encoder

class EncoderTester(object):
    def __init__(self):
        self.opt = get_parse_encoder()

        if torch.cuda.is_available():
            device = torch.device("cuda:" + self.opt.gpu_ids[0])
            self.opt.use_gpu = True
        else:
            device = torch.device("cpu")
            self.opt.use_gpu = False
        self.device = device

    def _test_reay(self):
        opt = self.opt
        # load the training config to update opt
        config_path = os.path.join("..",os.path.dirname(opt.checkpoint) + os.sep + 'opts.yaml')  # todo: add opt.checkpoint

        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        for group_dict_key, group_dict in config.items():
            if group_dict_key == 'Network Settings':
                for cfg, value in group_dict.items():
                    setattr(opt, cfg, value)
            else:
                for cfg, value in group_dict.items():
                    if not hasattr(opt, cfg):
                        setattr(opt, cfg, value)

        from train_img_encoder.nets_taskflow import make_model
        self.model = make_model(self.opt)
        if self.opt.use_gpu:
            self.model = self.model.to(self.device)
        # model.load_state_dict(torch.load(opt.checkpoint)) #org version
        checkpoint = torch.load(opt.checkpoint)
        self.model.load_state_dict(checkpoint["model_state"]) if "model_state" in checkpoint else self.model.load_state_dict(checkpoint)
        # self.model.eval()
        # self.dataloader = make_dataloader(opt, stage='test') if not hasattr(self, 'dataloader') else self.dataloader
        # return self.model,self.dataloader
        for param in self.model.parameters():
            param.requires_grad = False

from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import geo_trans
class SatMapDataset(Dataset):
    def __init__(self,
                 p_satinfo_json,
                 satimgsize2clip=224,
                 satimgsize2net=224,
                 rand_rot_sat=False,
                 **kwargs,
                 ):
        #1.read corresponding satellite image & mate info
        with open(p_satinfo_json, "r") as f:
            sat_infodict = json.load(f)
        self.satinfo_dict = sat_infodict
        self.geo_transform = self.satinfo_dict['geo_transform']
        # version 0,read tif as np:
        # self.sat_img = cv2.imread(self.satinfo_dict['tif_path']).astype(np.float32)[...,::-1] / 255.
        #  self.tif_h = self.sat_img.shape[0]
        #  self.tif_w = self.sat_img.shape[1]
        # version 1,read tif as Image:
        self.sat_img = Image.open(self.satinfo_dict['tif_path']) #hwc
        self.tif_h = self.sat_img.height
        self.tif_w = self.sat_img.width
        self.tif_hw_max = np.max([ self.tif_h, self.tif_w])

        #2.set other vals about satimg
        #  define the normed_halfimg_radius_rc that means positive samples
        self.satimgsize2net = satimgsize2net
        self.halfimg_radius_rc = self.satimgsize2net // 2. / self.tif_hw_max #about 30m
        self.halfimg_radius_meter = self.get_halfimg_radius_meter()

        #  define the vals about how to clip
        self.satimgsize2clip = satimgsize2clip
        self.img_edge_pixs = satimgsize2clip
        self.row_min_normalized = self.img_edge_pixs / self.tif_hw_max
        self.col_min_normalized = self.row_min_normalized
        self.row_max_normalized = ( self.tif_h - self.img_edge_pixs) / self.tif_hw_max
        self.col_max_normalized = ( self.tif_w - self.img_edge_pixs) / self.tif_hw_max
        self.row_width_normed = self.row_max_normalized -  self.row_min_normalized
        self.col_width_normed = self.col_max_normalized -  self.col_min_normalized

        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.satinfo_dict['mean'], std=self.satinfo_dict['std']),
        ]
        transform = transforms.Compose(transforms_list)
        self.sat_img_tensor = transform(self.sat_img)
        self.set_sat_transform()

    def get_halfimg_radius_meter(self):
        diff_lat =  self.satimgsize2net // 2. * np.abs(self.geo_transform[-1])
        diff_lon =  self.satimgsize2net // 2. * self.geo_transform[1]
        diff_met_lat = geo_trans.diff_lat_to_meter(diff_lat)
        diff_met_lon = geo_trans.diff_lon_to_meter(diff_lon,self.geo_transform[3])
        meter_radius = 0.5*(diff_met_lon+diff_met_lat)
        return meter_radius

    def clip_satimg_fm_rc(self, rc, type = 'tensor'):
        row = int(rc[0]*self.tif_hw_max)
        col = int(rc[1]*self.tif_hw_max)
        col_begin = col - self.satimgsize2net / 2
        col_end = col + self.satimgsize2net / 2
        row_begin = row - self.satimgsize2net / 2
        row_end = row + self.satimgsize2net / 2

        if type =='tensor':
            sat_img = self.sat_img_tensor[:, int(row_begin):int(row_end),int(col_begin):int(col_end)]  # chw for sat_img_tensor
        else:
            # sat_img = self.sat_img[int(row_begin):int(row_end),int(col_begin):int(col_end),:]
            sat_img = self.sat_img.crop((int(col_begin),int(row_begin),int(col_end),int(row_end)))
        return sat_img

    def mk_a_randrc(self):
        r = self.row_min_normalized + np.random.rand()* self.row_width_normed
        c = self.col_min_normalized + np.random.rand()* self.col_width_normed
        return np.array([r,c],dtype=np.float32)

    def set_sat_transform(self,random_rot=False):
        """
        sat_transform for training and testing sets are the same
        """
        if self.satimgsize2clip != 224 :
            transforms_list = [ transforms.Resize((224,224), interpolation=3) ]
        else:
            transforms_list = []

        if random_rot:
            transforms_list.append(transforms.RandomRotation(180))

        transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.satinfo_dict['mean'], std=self.satinfo_dict['std']),
        ]

        self.sat_transform = transforms.Compose(transforms_list)

    def mk_coord_grid(self,ovrelap=0.5,random=False,dtype=torch.float32):
        if not hasattr(self, 'meshgrid'):
            n_grid_h = int(self.tif_h / ((1-ovrelap)*self.satimgsize2clip))
            n_grid_w = int(self.tif_w / ((1-ovrelap)*self.satimgsize2clip))
            y_delta =  self.row_width_normed * 1/n_grid_h
            y_coords = self.row_min_normalized + y_delta * torch.arange(n_grid_h,dtype=dtype) + 0.5*y_delta
            x_delta = self.col_width_normed * 1/n_grid_w
            x_coords = self.col_min_normalized + x_delta * torch.arange(n_grid_w,dtype=dtype) + 0.5*x_delta
            xx, yy = torch.meshgrid(x_coords, y_coords, indexing='xy')
            self.meshgrid = torch.stack([yy.T, xx.T]).T
            self.y_delta = y_delta
            self.x_delta = x_delta
            self.n_grid_h = n_grid_h
            self.n_grid_w = n_grid_w

        if random:
            rand_delta = (torch.rand(self.meshgrid.shape,dtype=dtype)-0.5)*torch.tensor([self.y_delta,self.x_delta],dtype=dtype)
            gred2ret = rand_delta + self.meshgrid
            return gred2ret
        else:
            return self.meshgrid


    def __getitem__(self,index):
        sat_rc_rand = self.mk_a_randrc()
        satimg_rand = self.sat_transform(self.clip_satimg_fm_rc(sat_rc_rand,type='np'))
        return sat_rc_rand, satimg_rand

    def __len__(self):
        return  int(( self.tif_h* self.tif_w)/(self.satimgsize2clip**2) * 100)



def get_args():
    parser = argparse.ArgumentParser(description="Image benchmark using PyTorch bindings.")
    parser.add_argument('--p_satinfo_json',
                        default='/home/data/zwk/data_uavimgs_XianganXmu__512h_lineClassed/dataset_xmu_meta/satimgs_xiangan_xmu_info.json',
                        type=str, help='training dir path')
    parser.add_argument("n_epoch", nargs="?", type=int, default=1000, help="Number of training steps")
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
    parser.add_argument('--num_worker', default=16, type=int, help='batchsize')
    parser.add_argument('--tensorboard', action='store_true', default = True)
    parser.add_argument('--exp_name', default='debug', type=str)



    return parser.parse_args()

import yaml
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
if __name__ == "__main__":
    torch.manual_seed(666)
    # load the trained encoder
    img_encoder = EncoderTester()
    img_encoder._test_reay()

    # prepare the dataset
    args=get_args()
    dataset = SatMapDataset(p_satinfo_json = '/home/data/zwk/data_uavimgs_XianganXmu__512h_lineClassed/dataset_xmu_meta/satimgs_xiangan_xmu_info.json',
                            rand_rot_sat = False,
                            )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize,num_workers=args.num_worker,
                                                 pin_memory=True,shuffle=True,drop_last=True)
    dataloader.dataset.mk_coord_grid()

    device = torch.device("cuda:0")
    map_encoder_cfg = {
		"otype": "CutlassMLP",#CutlassMLP or FullyFusedMLP
		"activation": "ReLU",
		"output_activation": "None",
		"n_neurons": 512,
		"n_hidden_layers": 4
	}
    optimizer_cfg= {
        "otype": "Adam",
        "lr": 1e-2,
        "beta1": 0.9,
        "beta2": 0.99,
        "eps": 1e-8,
        "l2_reg": 1e-8
    }

    # config tensorborad
    writer = SummaryWriter(f"/home/data/zwk/pyproj_DUAV_salad_6.4/train_mlp_map/exps/{args.exp_name}/train_tensorboard.log") if args.tensorboard else None

    # dataloader.dataset.mk_coord_grid()
    # map_encoder = tcnn.Network(n_input_dims=512+2, n_output_dims=1,network_config=map_encoder_cfg)
    from pos_encoder import PositionalEncoder
    pos_encoder = PositionalEncoder(multires=10,input_dims=2)

    from mlp_creater import create_mlp,init_weights
    dims = [512+pos_encoder.out_dim,512,512,1]
    activation = nn.LeakyReLU
    map_encoder = create_mlp(dims, activation_fn=activation, norm_type='layer').to(device)
    map_encoder.apply(lambda m: init_weights(m, method='kaiming', nonlinearity='leaky_relu'))

    optimizer = torch.optim.Adam(map_encoder.parameters(), lr=optimizer_cfg['lr'],betas=(optimizer_cfg['beta1'],optimizer_cfg['beta2']),
                                 eps=optimizer_cfg['eps'],weight_decay=optimizer_cfg['l2_reg'])
    criterion = torch.nn.CrossEntropyLoss()
    gaussian_dist = torch.distributions.Normal(loc=0, scale=1.)

    # for epoch in range(args.n_epoch):
    #     for it, data in tqdm(enumerate(dataloader)):
    #         rcs, imgs = data
    #         feats = img_encoder.model(imgs.to(device))
    #         for i,feat in enumerate(feats):
    #             grid = dataloader.dataset.mk_coord_grid(random=True,dtype=torch.float32)
    #             rc2input = torch.concatenate([rcs[i].unsqueeze(0),grid.reshape(-1,2)],dim=0)
    #             rc_encoded = pos_encoder(rc2input)
    #             input = torch.concatenate([rc_encoded.to(device), feat.expand(rc2input.shape[0],-1)], -1)
    #             output = map_encoder.forward(input)


    for epoch in range(args.n_epoch):
        for it, data in tqdm(enumerate(dataloader)):
            rcs, imgs = data
            feats = img_encoder.model(imgs.to(device))
            for i,feat in enumerate(feats):
                grid = dataloader.dataset.mk_coord_grid(random=True,dtype=torch.float32)
                rc2input = torch.concatenate([rcs[i].unsqueeze(0),grid.reshape(-1,2)],dim=0)
                rc_encoded = pos_encoder(rc2input)
                input = torch.concatenate([rc_encoded.to(device), feat.expand(rc2input.shape[0],-1)], -1)
                output = map_encoder.forward(input)

                # #kl_loss
                dist = torch.norm(rcs[i]-rc2input,dim=-1)
                rel_dist = dist / dataloader.dataset.halfimg_radius_rc
                gaussian_pdf = torch.exp(gaussian_dist.log_prob(rel_dist)).to(device)
                gaussian_pdf = gaussian_pdf / torch.sum(gaussian_pdf)
                logits = torch.softmax(output.squeeze(), dim=-1)
                kl_loss = -gaussian_pdf*torch.log(logits)
                # loss = kl_loss[0]+kl_loss[1:].mean()
                loss = kl_loss.sum()*0.1

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




