import torch
import yaml
import argparse
import os
from typing import Dict, Any

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    递归地将一个嵌套字典“拍平”。
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def yaml_to_argparse(yaml_path: str) -> argparse.ArgumentParser:
    """
    从指定的YAML文件路径读取内容，并创建一个argparse.ArgumentParser对象。

    :param yaml_path: YAML配置文件的路径。
    :return: 一个配置好所有参数和默认值的ArgumentParser对象。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"指定的YAML文件不存在: {yaml_path}")

    # 2. 使用PyYAML从文件加载YAML内容
    with open(yaml_path, 'r', encoding='utf-8') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"解析YAML文件时出错: {e}")
            return argparse.ArgumentParser()

    # 3. 将嵌套的配置字典拍平
    flat_config = flatten_dict(config)

    # 4. 创建ArgumentParser对象
    parser = argparse.ArgumentParser(
        description="从YAML文件加载配置。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 5. 遍历拍平后的字典，为每个键值对添加一个参数
    for key, value in flat_config.items():
        arg_name = f'--{key}'
        value_type = type(value)

        if value_type == bool:
            parser.add_argument(
                arg_name,
                type=lambda x: (str(x).lower() == 'true'),
                default=value,
                help=f"从YAML加载: {key}"
            )
        elif value_type == list:
            parser.add_argument(
                arg_name,
                nargs='*',
                default=value,
                help=f"从YAML加载: {key}"
            )
        else:
            parser.add_argument(
                arg_name,
                type=value_type if value is not None else str,
                default=value,
                help=f"从YAML加载: {key}"
            )

    return parser


class ImgEncoder(object):
    def __init__(self,config_path,ckpt2load_path):

        parser = yaml_to_argparse(config_path)
        # 4. 解析参数 (这会加载文件中的默认值)
        args = parser.parse_args()
        self.opt = args
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            self.opt.use_gpu = True
        else:
            device = torch.device("cpu")
            self.opt.use_gpu = False
        self.device = device

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

        from models.taskflow import make_model
        self.model = make_model(self.opt)
        if self.opt.use_gpu:
            self.model = self.model.to(self.device)
        # model.load_state_dict(torch.load(opt.checkpoint)) #org version
        checkpoint = torch.load(ckpt2load_path)
        self.model.load_state_dict(checkpoint["model_state"]) if "model_state" in checkpoint else self.model.load_state_dict(checkpoint) #todo:mk checkpoint_path as a parameter
        # self.model.eval()
        # self.dataloader = make_dataloader(opt, stage='test') if not hasattr(self, 'dataloader') else self.dataloader
        # return self.model,self.dataloader
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def get_output_dim(self):
        opt = self.opt
        if opt.head == '':
            if opt.backbone == 'ViTS-224':
                return 384
            else:
                return 768
        elif opt.head == 'FSRA':
            return opt.num_bottleneck * (opt.block+1)



if __name__ == '__main__':
    img_encoder = ImgEncoder()
    img_encoder._test_reay()
