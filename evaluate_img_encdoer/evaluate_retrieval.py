import scipy.io
import torch
import numpy as np
import os


######################################################################
if __name__ == "__main__":
    path2res_mat = '/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug/epoch003_overlap0.75_radius32m.mat'
    gpu_id = int(0)
    torch.cuda.set_device(gpu_id)

    result = scipy.io.loadmat(path2res_mat)
    query_feature = torch.FloatTensor(result['query_feat'])
    query_rc = result['query_rc']
    query_latlon = result['query_latlon']
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_feat'])
    gallery_rc = result['gallery_rc']
    gallery_latlon = result['gallery_latlon']
    # gallery_sg = result['sg_feat']

    from eval_recall_fm_salad import compute_recall_from_feat
    d = compute_recall_from_feat(query_feature.contiguous(),gallery_feature.contiguous(),query_label,[1,5,20,50,200],faiss_gpu=False)
    file2write = f"results_{os.path.basename(path2res_mat).split('.mat')[0]}.txt"
    path2file = os.path.join(os.path.dirname(path2res_mat),file2write)
    with open(path2file, "w") as F:
        info = "Recall"
        for k,v in d.items():
            info = info + f" @{k}:{v*100:.2f} "
        F.write(info)



