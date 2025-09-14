import scipy.io
import torch
import numpy as np
import os
from train_img_encoder.util_circorr_fm_radon import norm_rot

########################### from F3Loc ###########################
def transit(
    prob_vol,
    transition,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
):
    """
    Input:
        prob_vol: torch.tensor(H, W, O), probability volume before the transition
        transition: ego motion
        sig_o: stddev of rotation
        sig_x: stddev in x translation
        sig_w: stddev in y translation
        tsize: translational filter size
        rsize: rotational filter size
        resolution: resolution of the grid [m/pixel]
    """
    H, W, O = list(prob_vol.shape)
    # construction O filters
    filters_trans, filter_rot = get_filters(
        transition,
        O,
        sig_o=sig_o,
        sig_x=sig_x,
        sig_y=sig_y,
        tsize=tsize,
        rsize=rsize,
        resolution=resolution,
        known_xy=False,
    )  # (O, 5, 5), (5,)

    # set grouped 2d convolution, O as channels
    prob_vol = prob_vol.permute((2, 0, 1))  # (O, H, W)

    # convolve with the translational filters
    # NOTE: make sure the filter is convolved correctly need to flip
    prob_vol = F.conv2d(
        prob_vol,
        weight=filters_trans.unsqueeze(1).flip([-2, -1]),
        bias=None,
        groups=O,
        padding="same",
    )  # (O, H, W)

    #debug for vis:
    # from vis_featmap import vis_multi
    # vis_multi(prob_vol.detach().cpu().numpy(),p2save='/home/data/zwk/pyproj_f3loc/prior_transed_t0_linefilter_kszise15_distau1._gswith0.1.jpg',camp='jet',fighw=(4,2.2))
    # vis_multi(filters_trans.unsqueeze(1).flip([-2, -1]).squeeze().detach().cpu().numpy(),p2save='/home/data/zwk/pyproj_f3loc/linefilter_trans_filped_t0_wxydirect_distau0.5_gswith0.1.jpg',camp='jet')
    # vis_multi(filters_trans.squeeze().detach().cpu().numpy(),p2save='/home/data/zwk/pyproj_f3loc/linefilter_trans_t0_wxydirect_distau0.5_gswith0.1.jpg',camp='jet')

    # convolve with rotational filters
    # reshape as batch
    prob_vol = prob_vol.permute((1, 2, 0))  # (H, W, O)
    prob_vol = prob_vol.reshape((H * W, 1, O))  # (HxW, 1, O)
    prob_vol = F.pad(
        prob_vol, pad=[int((rsize - 1) / 2), int((rsize - 1) / 2)], mode="circular"
    )
    prob_vol = F.conv1d(
        prob_vol, weight=filter_rot.flip(dims=[-1]).unsqueeze(0).unsqueeze(0), bias=None
    )  # TODO (HxW, 1, O)

    # reshape
    prob_vol = prob_vol.reshape([H, W, O])  # (H, W, O)
    # normalize
    prob_vol = prob_vol / prob_vol.sum()

    return prob_vol


def get_filters(
    O=36,
    sig_o=0.1,
    sig_x=0.05,
    sig_y=0.05,
    tsize=5,
    rsize=5,
    resolution=0.1,
    transition=None,
    known_xy=True,
):
    """
    Return O different filters according to the ego-motion
    Input:
        transition: torch.tensor (3,), ego motion
    Output:
        filters_trans: torch.tensor (O, 5, 5)
                    each filter is (fH, fW)
        filters_rot: torch.tensor (5)
    """
    # NOTE: be careful about the orienation order, what is the orientation of the first layer?

    # get the filters according to gaussian
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=transition.device),
    )
    # add units
    grid_x = grid_x * resolution  # 0.1m
    grid_y = grid_y * resolution  # 0.1m

    # calculate center of the gaussian for 36 orientations
    # center for orientation stays the same
    center_o = transition[-1]
    # center_x and center_y depends on the orientation, in total O different, rotate
    orns = (
        torch.arange(0, O, dtype=torch.float32, device=transition.device)
        / O
        * 2
        * torch.pi
    )  # (O,)

    if known_xy:
        c_th = torch.cos(orns).reshape((O, 1, 1))  # (O, 1, 1)
        s_th = torch.sin(orns).reshape((O, 1, 1))  # (O, 1, 1)
        center_x = transition[0] * c_th - transition[1] * s_th  # (O, 1, 1)
        center_y = transition[0] * s_th + transition[1] * c_th  # (O, 1, 1)

        # add uncertainty
        filters_trans = torch.exp(
            -((grid_x - center_x) ** 2) / (sig_x**2) - (grid_y - center_y) ** 2 / (sig_y**2)
        )  # (O, 5, 5)
        # normalize
        filters_trans = filters_trans / filters_trans.sum(-1).sum(-1).reshape((O, 1, 1))
    else:
        known_xy_dir = True
        if known_xy_dir:
            direction_angle = torch.arctan2(transition[1], transition[0])
            direction_angle = direction_angle + orns

            cos_theta = torch.cos(direction_angle)
            sin_theta = torch.sin(direction_angle)
            dist_tau = 1.
            sigomid_slope = 200
            # 分量 a): 距离约束 (Sigmoid)
            dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)
            sigmoid_part = 1 / (1 + torch.exp(sigomid_slope * (dist - dist_tau)))
            # 分量 b): 方向约束 (垂直距离的高斯衰减)
            # 计算点 (grid_x, grid_y) 到方向 theta 所在直线的垂直距离的平方
            # d_perp = |-x*sin + y*cos|
            gaussian_std = 0.1
            d_perp_sq = (-grid_x.unsqueeze(0) * sin_theta.reshape(-1,1,1) + grid_y.unsqueeze(0) * cos_theta.reshape(-1,1,1)) ** 2
            gaussian_part = torch.exp(-d_perp_sq / (2 * gaussian_std ** 2))
            # 3. 结合两个约束 (逐元素相乘)
            kernel = sigmoid_part * gaussian_part
            filters_trans = kernel / kernel.sum(dim=(-2,-1)).reshape((O, 1, 1))
            # filters_trans2vis = filters_trans*6+filters_trans_org
            # vis_multi(filters_trans2vis.detach().cpu().numpy(),'/home/data/zwk/pyproj_f3loc/line6x&point_kernels_wxydirect_distau0.6_gswith0.1.jpg')
        else:
            dist = torch.sqrt(grid_x**2 + grid_y**2)
            dist_tau = 0.15
            sigomid_slope = 200
            sigomid_distb= 1 / (1 + torch.exp(sigomid_slope * (dist - dist_tau)))
            # vis_single(sigomid_distb.detach().cpu().numpy(),'/home/data/zwk/pyproj_f3loc/sigmoid_distb.jpg',camp='coolwarm')
            sigomid_distb = (sigomid_distb / sigomid_distb.sum(-1).sum(-1)).unsqueeze(0).repeat(O, 1, 1)
            filters_trans = sigomid_distb

    # rotation filter
    grid_o = (
        torch.arange(-(rsize - 1) / 2, (rsize + 1) / 2, 1, device=transition.device)
        / O
        * 2
        * torch.pi
    )
    filter_rot = torch.exp(-((grid_o - center_o) ** 2) / (sig_o**2))  # (5)

    return filters_trans, filter_rot

########################### Changes based on F3Loc ###########################
def get_trans_filters(
        ego=0,
        tsize=5,
        O=36,
        resolution=1,
        # dist_tau=1,
        sigomid_slope=10,
        device=torch.device('cpu'),
):
    """
    tsize: kernal size
    resolution:resolution=1即以像素为单位
    """
    # get the filters according to gaussian
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=device),
    )
    # add units
    grid_x = grid_x * resolution  # 0.1m
    grid_y = grid_y * resolution  # 0.1m

    # mk orientation for filters
    orns = (
        torch.arange(0, O, dtype=torch.float32, device=device)
        / O
        * 2
        * torch.pi
    )  # (O,)
    orns = orns + ego
    cos_theta = torch.cos(orns)
    sin_theta = torch.sin(orns)

    dist_tau = tsize//2 + 1
    shift_d = tsize//2 - 1
    # 1. 计算每个方向上的中心偏移量
    #    cos_theta 和 sin_theta 的 shape 为 (O,)，通过 reshape 适配广播机制
    center_x = shift_d * cos_theta.reshape(-1, 1, 1)
    center_y = shift_d * sin_theta.reshape(-1, 1, 1)

    # 2. 分量 a): 距离约束 (Sigmoid)，现在计算网格点到 *偏移后中心* 的距离
    #    grid_x/grid_y 的 shape 为 (tsize, tsize)，通过 unsqueeze 适配广播机制
    dist = torch.sqrt((grid_x.unsqueeze(0) - center_x) ** 2 + (grid_y.unsqueeze(0) - center_y) ** 2)
    sigmoid_part = 1 / (1 + torch.exp(sigomid_slope * -(dist_tau-dist) ))

    # 分量 b): 方向约束 (垂直距离的高斯衰减)
    # 计算点 (grid_x, grid_y) 到方向 theta 所在直线的垂直距离的平方
    # d_perp = |-x*sin + y*cos|
    gaussian_std = resolution*1.5
    d_perp_sq = (-grid_x.unsqueeze(0) * sin_theta.reshape(-1, 1, 1) + grid_y.unsqueeze(0) * cos_theta.reshape(-1, 1, 1)) ** 2
    gaussian_part = torch.exp(-d_perp_sq / (2 * gaussian_std ** 2))
    # 3. 结合两个约束 (逐元素相乘)
    kernel = sigmoid_part * gaussian_part
    filters_trans = kernel / kernel.sum(dim=(-2, -1)).reshape((O, 1, 1))
    return filters_trans

def get_trans_filters_v0(
        ego=0,
        tsize=5,
        O=36,
        resolution=1,
        dist_tau=0,
        sigomid_slope=10,
        device=torch.device('cpu'),
):
    """version 0
    tsize: kernal size
    resolution:
    """
    # get the filters according to gaussian
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=device),
    )
    # add units
    grid_x = grid_x * resolution  # 0.1m
    grid_y = grid_y * resolution  # 0.1m

    # mk orientation for filters
    orns = (
            torch.arange(0, O, dtype=torch.float32, device=device)
            / O
            * 2
            * torch.pi
    )  # (O,)
    orns = orns + ego
    cos_theta = torch.cos(orns)
    sin_theta = torch.sin(orns)

    # mk line filters
    sigomid_slope = sigomid_slope
    # 分量 a): 距离约束 (Sigmoid)
    dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    sigmoid_part = 1 / (1 + torch.exp(sigomid_slope * (dist - (tsize+1))))

    # 分量 b): 方向约束 (垂直距离的高斯衰减)
    # 计算点 (grid_x, grid_y) 到方向 theta 所在直线的垂直距离的平方
    # d_perp = |-x*sin + y*cos|
    gaussian_std = resolution*1.2
    d_perp_sq = (-grid_x.unsqueeze(0) * sin_theta.reshape(-1, 1, 1) + grid_y.unsqueeze(0) * cos_theta.reshape(-1, 1,
                                                                                                              1)) ** 2
    gaussian_part = torch.exp(-d_perp_sq / (2 * gaussian_std ** 2))
    # 3. 结合两个约束 (逐元素相乘)
    kernel = sigmoid_part * gaussian_part
    filters_trans = kernel / kernel.sum(dim=(-2, -1)).reshape((O, 1, 1))
    return filters_trans


def get_rot_fitler(
        O=72,
        rotdeg_max = 75,
        trans_rotrad_anticlock = 0,
        sig_o = 0.1,
):
    rsize = np.floor(rotdeg_max/(360/O))
    grid_o = (
        torch.arange(-(rsize - 1) / 2, (rsize + 1) / 2, 1, device=device)
        / O
        * 2
        * torch.pi
    )
    filter_rot = torch.exp(-((grid_o - trans_rotrad_anticlock) ** 2) / (sig_o**2))
    return filter_rot



import torch.nn.functional as F
######################################################################
if __name__ == "__main__":
    path2res_mat = '/home/data/zwk/pyproj_DUAV_salad_6.4/exps/debug_zuchwil/epoch015_overlap0.5_radius62m.mat'
    gpu_id = int(0)
    device = torch.device('cuda:%d' % gpu_id)

    result = scipy.io.loadmat(path2res_mat)
    query_feature = torch.FloatTensor(result['query_feat'])
    query_rc = result['query_rc']
    query_latlon = result['query_latlon']
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_feat'])
    gallery_rc = result['gallery_rc']
    gallery_latlon = result['gallery_latlon']
    gallery_hw = result['gallery_hw']
    gallery_sg = torch.FloatTensor(result['gallery_sg'])
    query_sg = torch.FloatTensor(result['query_sg'])
    rotdeg_fm_north_anticlock = result['rotdeg_fm_north_anticlock'][0]
    rotdeg_fm_north_anticlock_positive = rotdeg_fm_north_anticlock[-query_feature.shape[0]:]
    relrot_normed = norm_rot(rotdeg_fm_north_anticlock_positive)

    from train_img_encoder.util_circorr_fm_radon import circorr_fm_radon
    gallery_feature = gallery_feature.to(device)
    query_feature = query_feature.to(device)
    gallery_sg = gallery_sg.to(device)
    query_sg = query_sg.to(device)

    #transit by filters
    from vis_featmap import vis_single_featmap_with_direct
    O = 72
    # tsize = 13
    # trans_filters = get_trans_filters(tsize=tsize,O=O,device=device)
    # vis_multi_featmap(trans_filters)

    # import torchvision.transforms as T
    # kernel_size = 9
    # sigma = 5.0
    # gaussian_blur_transformer = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    for i,q_feat in enumerate(query_feature):

        # 预测：
        if i == 0:
            bel_ti_ = torch.ones(gallery_feature.shape[0], O, device=device)
        else:
            bel_ti_ = bel_ti.reshape(*gallery_hw[0],-1).permute(2,0,1)

            #debug:
            prob_dist, orientations = torch.max(bel_ti_, dim=0)
            pred_y, pred_x = torch.where(prob_dist == prob_dist.max())
            orn = orientations[pred_y, pred_x]*(360/O)
            pred_xyr = torch.tensor([pred_x, pred_y, orn]).cpu().numpy()
            gtxy = (query_rc[i]*np.max(prob_dist.shape))[::-1]
            gtr = relrot_normed[i]
            gt_xyr = np.array([gtxy[0],gtxy[1], gtr])
            p2save = os.path.join('exps/debug_vis',f'{i}.png')
            # vis_single_featmap_with_direct(prob_dist,p2save=p2save,title=f'{i}th',xyrot1=pred_xyr,xyrot2=gt_xyr,camp='coolwarm')
            vis_single_featmap_with_direct(prob_dist,p2save=None,title=f'{i}th',xyrot1=pred_xyr,xyrot2=gt_xyr,camp='coolwarm')

            # xy_transit:
            O = 72
            tsize = 13
            trans_filters = get_trans_filters_v0(
                ego = np.deg2rad(relrot_normed[i]-relrot_normed[i-1]),
                tsize= tsize,
                O=O,
                device=device
            )
            bel_ti_ = F.conv2d(
                bel_ti_,
                weight=trans_filters.unsqueeze(1).flip([-2, -1]),
                bias=None,
                groups=trans_filters.shape[0],
                padding="same",
            )  # (O, H, W)

            # rot_transit
            rotdeg_threshold=75
            rot_filter = get_rot_fitler(
                O=O,
                rotdeg_max = rotdeg_threshold,
                trans_rotrad_anticlock = np.deg2rad(relrot_normed[i]-relrot_normed[i-1]),
                sig_o = 0.1,
            )
            rsize = np.floor(rotdeg_threshold / (360 / O))
            bel_ti_ = bel_ti_.reshape(bel_ti_.shape[0],1,-1).permute(2,1,0)  # (HxW, 1, O)
            bel_ti_ = F.pad(
                bel_ti_, pad=[int((rsize - 1) / 2), int((rsize - 1) / 2)], mode="circular"
            )
            bel_ti_ = F.conv1d(
                bel_ti_, weight=rot_filter.flip(dims=[-1]).unsqueeze(0).unsqueeze(0), bias=None
            ).squeeze()  # (HxW, 1, O)

        if i==117:
            print('debug')
            #使用 imageio 将所有帧合成为视频
            import imageio
            output_video_path = os.path.join('../train_img_encoder/exps', 'debug3_wo_dirtinfo' + '.mp4')
            frame_filenames = os.listdir('exps/debug_vis')
            frame_filenames.sort(key=lambda x: int(x.split('.')[0]))
            frame_filenames = [os.path.join('exps/debug_vis',name) for name in frame_filenames]
            with imageio.get_writer(output_video_path, fps=3) as writer:
                for filename in frame_filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)

        #单次观测
        q_feat = query_feature[i]
        xy_distb = gallery_feature @ q_feat.unsqueeze(1)
        xy_distb_normed = (xy_distb-xy_distb.min()) / (xy_distb.max()-xy_distb.min()+1e-5)
        # xy_distb_normed = torch.exp(xy_distb_normed)
        # xy_distb_normed = xy_distb_normed /xy_distb_normed.max()
        # xy_distb_normed = xy_distb / xy_distb.sum() * 1e5

        # 加入radon旋转观测信息
        # circor = circorr_fm_radon(query_sg[0][None, None, ...], gallery_sg.unsqueeze(1))
        # circor_normed = (circor-circor.min()) / (circor.max()-circor.min()+1e-5)
        # circor_normed = torch.exp(circor_normed)
        # circor_normed = circor_normed /circor_normed.max()
        # circor_normed = circor / circor.sum(dim=1, keepdim=True)
        # prior = xy_distb_normed * circor_normed  # prior


        prior = xy_distb_normed.repeat(1,O)
        # prior = xy_distb_normed * torch.ones(gallery_feature.shape[0], O, device=device)
        # prior = prior / prior.sum()*1e5

        #基于观测的更新：
        beta = 0.1
        # bel_ti_ = bel_ti_/bel_ti_.sum()*1e5
        if i==0:
            bel_ti = prior
        else:
            bel_ti_ = (bel_ti_-bel_ti_.min())/(bel_ti_.max()-bel_ti_.min()+1e-5) if i>0 else bel_ti_
            # bel_ti_ = torch.exp(bel_ti_/0.8)
            # bel_ti_ = bel_ti_/bel_ti_.max()
            # bel_ti = beta*(bel_ti_*prior) + (1-beta)*prior
            bel_ti = prior * bel_ti_

        n_nan = torch.isnan(bel_ti).sum().item()
        print(f'{i}th; n_nan={n_nan}')



    # from vis_featmap import vis_rot_func
    # prob_1d = prob_vol[:,50,50]
    # O=72
    # rsize = np.floor(75/(360/O))
    # grid_o = (
    #     torch.arange(-(rsize - 1) / 2, (rsize + 1) / 2, 1, device=device)
    #     / O
    #     * 2
    #     * torch.pi
    # )
    # center_o = np.deg2rad(30)
    # sig_o=0.1
    # filter_rot = torch.exp(-((grid_o - center_o) ** 2) / (sig_o**2))  # (5)
    # # vis_rot_func(filter_rot.detach().cpu().numpy(),np.rad2deg(grid_o.detach().cpu().numpy()),'/Localize/hsj/zwk/filter_rot.jpg',vis_mode='flat')
    # # vis_rot_func(prob_1d.detach().cpu().numpy(),[i for i in range(O)],'/Localize/hsj/zwk/prob_1d.jpg',vis_mode='flat')
    #
    # prob_1d = prob_1d[None,None,...]
    # prob_1d = F.pad(
    #     prob_1d, pad=[int((rsize - 1) / 2), int((rsize - 1) / 2)], mode="circular"
    # )
    # prob_1d = F.conv1d(
    #     prob_1d, weight=filter_rot.flip(dims=[-1]).unsqueeze(0).unsqueeze(0), bias=None
    # )  # TODO (HxW, 1, O)
    # vis_rot_func(prob_1d.squeeze().detach().cpu().numpy(),[i for i in range(O)],'/Localize/hsj/zwk/prob_1d_conved.jpg',vis_mode='flat')
    #
    # from vis_featmap import vis_multi_featmap
    # vis_multi_featmap(trans_filters.detach().cpu().numpy(),'/Localize/hsj/zwk/linefilters.jpg')

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


