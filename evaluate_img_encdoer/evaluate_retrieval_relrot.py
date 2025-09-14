import scipy.io
import torch
import numpy as np
import os
from train_img_encoder.util_circorr_fm_radon import norm_rot
######################################################################
if __name__ == "__main__":
    path2res_mat = 'exps/debug/epoch003_overlap0.75_radius32m.mat'
    gpu_id = int(6)
    torch.cuda.set_device(gpu_id)

    result = scipy.io.loadmat(path2res_mat)
    query_feature = torch.FloatTensor(result['query_feat'])
    query_rc = result['query_rc']
    query_latlon = result['query_latlon']
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_feat'])
    gallery_rc = result['gallery_rc']
    gallery_latlon = result['gallery_latlon']
    gallery_sg = result['gallery_sg']
    query_sg = result['query_sg']
    rotdeg_fm_north_anticlock = result['rotdeg_fm_north_anticlock'][0]
    rotdeg_fm_north_anticlock_positive = rotdeg_fm_north_anticlock[-query_feature.shape[0]:]
    relrot_normed = norm_rot(rotdeg_fm_north_anticlock_positive)

    # test the relrot
    from train_img_encoder.util_circorr_fm_radon import circorr_fm_radon
    nearest_satlabel = [labels[0][0] for labels in query_label]
    positive_sg = gallery_sg[nearest_satlabel]
    circor = circorr_fm_radon(torch.tensor(query_sg).unsqueeze(1),torch.tensor(positive_sg).unsqueeze(1)).numpy()
    pred_rot = np.argmax(circor,axis=-1)*5
    deg_tau = 15.5
    deg_diff = np.abs(pred_rot - relrot_normed)
    recall_rot = ((deg_diff <= deg_tau).sum() + (deg_diff >= (360 - deg_tau)).sum()) / pred_rot.shape[0]
    file2write = f"recall_rot_{os.path.basename(path2res_mat).split('.mat')[0]}.txt"
    path2file = os.path.join(os.path.dirname(path2res_mat),file2write)
    with open(path2file, "w") as F:
        info = f"RecallRot@{deg_tau:.1f}={recall_rot*100:.2f}\n"
        F.write(info)
    #todo: debug with visualization

    # test the performce on retrieval
    from eval_recall_fm_salad import compute_recall_from_feat
    d = compute_recall_from_feat(query_feature.contiguous(),gallery_feature.contiguous(),query_label,[1,5,20,50,200],faiss_gpu=False,device_id=6)
    file2write = f"recall_{os.path.basename(path2res_mat).split('.mat')[0]}.txt"
    path2file = os.path.join(os.path.dirname(path2res_mat),file2write)
    with open(path2file, "w") as F:
        info = "Recall"
        for k,v in d.items():
            info = info + f" @{k}:{v*100:.2f} "
        F.write(info)


