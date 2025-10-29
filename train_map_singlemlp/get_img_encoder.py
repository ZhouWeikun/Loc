import torch
import yaml

from train_img_encoder.trainer_retrival_relrot import get_parse as get_parse_encoder
class ImgEncoder(object):
    def __init__(self):
        self.opt = get_parse_encoder()

        if torch.cuda.is_available():
            device = torch.device("cuda:" + self.opt.gpu_ids[0])
            self.opt.use_gpu = True
        else:
            device = torch.device("cpu")
            self.opt.use_gpu = False
        self.device = device

    def _test_reay(self,config_path,ckpt2load_path):
        opt = self.opt
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)
        for group_dict_key, group_dict in config.items():
            for cfg, value in group_dict.items():
                setattr(opt, cfg, value)
            # if group_dict_key == 'Network Settings':
            #     for cfg, value in group_dict.items():
            #         setattr(opt, cfg, value)
            # else:
            #     for cfg, value in group_dict.items():
            #         if not hasattr(opt, cfg):
            #             setattr(opt, cfg, value)

        from train_img_encoder.nets_taskflow import make_model
        self.model = make_model(self.opt)
        if self.opt.use_gpu:
            self.model = self.model.to(self.device)
        # model.load_state_dict(torch.load(opt.checkpoint)) #org version
        checkpoint = torch.load(ckpt2load_path)
        self.model.load_state_dict(checkpoint["model_state"]) if "model_state" in checkpoint else self.model.load_state_dict(checkpoint) #todo:mk checkpoint_path as a parameter
        # self.model.eval()
        # self.dataloader = make_dataloader(opt, stage='test') if not hasattr(self, 'dataloader') else self.dataloader
        # return self.model,self.dataloader
        for param in self.model.parameters():
            param.requires_grad = False

    def get_output_dim(self):
        opt = self.opt
        if opt.head == '':
            return 768
        elif opt.head == 'FSRA':
            return opt.num_bottleneck * (opt.block+1)

if __name__ == '__main__':
    img_encoder = ImgEncoder()
    img_encoder._test_reay()
