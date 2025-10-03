import tinycudann as tcnn
import json
import torch

with open('/home/data/zwk/pyproj_neuloc_v0/train_img_encoder/config_hash.json') as config_file:
    config = json.load(config_file)
# model = tcnn.NetworkWithInputEncoding(n_input_dims=4, n_output_dims=256, encoding_config=config["encoding"],
#                                       network_config=config["network"])
# model.jit_fusion = tcnn.supports_jit_fusion()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 1
encoding = tcnn.Encoding(n_input_dims=2, encoding_config=config["encoding"]).to(device)
batch = torch.rand([batch_size, 2], dtype=torch.float32,device=device)
output = encoding(batch)
