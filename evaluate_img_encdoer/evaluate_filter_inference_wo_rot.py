import scipy.io
import torch
import numpy as np
from train_img_encoder.util_circorr_fm_radon import norm_rot
from mertic_learning import LpDistance

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

########################### Changes based on F3Loc by zwk ###########################
def get_gaussian_filter(
        tsize: int,
        L: float,
        resolution: float = 1.0,
        device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    创建一个简化的、各向同性的高斯扩散滤波器。
    Args:
        tsize (int): 滤波器（卷积核）的边长，例如 11、21等，建议为奇数。
        L (float): 特征扩散尺度（半径）。这定义了扩散的主要范围。
        resolution (float): 每个像素代表的物理尺度，例如 0.1 (米/像素)。
        device (torch.device): 计算设备 (cpu or cuda)。

    Returns:
        torch.Tensor: 一个 [tsize, tsize] 大小的二维高斯滤波器，其元素总和为1。
    """
    # 1. 创建坐标网格
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=device),
        indexing='ij'  # 保证 y, x 顺序
    )

    # 将像素坐标转换为物理单位坐标
    grid_x = grid_x * resolution
    grid_y = grid_y * resolution

    # 2. 核心步骤：根据扩散尺度 L 计算高斯标准差 sigma
    # 我们设定 L 为 2 倍标准差，这覆盖了约95%的能量。
    # 这是一个常用且效果平滑的准则。
    # sigma = L / 2.0
    sigma = L / 2.0

    # 处理 L=0 的极端情况，此时 sigma=0，滤波器应为一个在中心点的脉冲
    if sigma == 0:
        kernel = torch.zeros((tsize, tsize), device=device)
        center_idx = tsize // 2
        kernel[center_idx, center_idx] = 1.0
        return kernel

    # 3. 计算二维高斯函数
    # 为了计算效率，先计算到中心点距离的平方
    dist_sq = grid_x ** 2 + grid_y ** 2

    # 应用高斯公式
    kernel = torch.exp(-dist_sq / (2 * sigma ** 2))

    # 4. 归一化
    # 确保滤波器的总和为1，这样在卷积时总概率保持不变
    kernel = kernel / kernel.sum()

    return kernel


def get_trans_filters_line(
        tsize=11,
        O=36,
        ego=0,
        gaussian_std=None,
        sigomid_slope=10,
        resolution=1,
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
    orns = (torch.arange(0, O, dtype=torch.float32, device=device) / O*torch.pi)  # (O,)
    orns = orns + ego
    cos_theta = torch.cos(orns)
    sin_theta = torch.sin(orns)

    # mk line filters
    sigomid_slope = sigomid_slope
    # 分量 a): 距离约束 (Sigmoid)
    dist = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    sigmoid_part = 1 / (1 + torch.exp(sigomid_slope * (dist - (tsize+2))))

    # 分量 b): 方向约束 (垂直距离的高斯衰减)
    # 计算点 (grid_x, grid_y) 到方向 theta 所在直线的垂直距离的平方
    # d_perp = |-x*sin + y*cos|
    gaussian_std = resolution*1.5 if gaussian_std is None else gaussian_std
    d_perp_sq = (-grid_x.unsqueeze(0) * sin_theta.reshape(-1, 1, 1) + grid_y.unsqueeze(0) * cos_theta.reshape(-1, 1,1)) ** 2
    gaussian_part = torch.exp(-d_perp_sq / (2 * gaussian_std ** 2))
    # 3. 结合两个约束 (逐元素相乘)
    kernel = sigmoid_part * gaussian_part
    filters_trans = kernel / kernel.sum(dim=(-2, -1)).reshape((O, 1, 1))
    return filters_trans


def get_trans_filters_halfline(
        tsize=5,
        O=36,
        ego=None,
        resolution=1,
        sigomid_slope=10,
        gaussian_std=None,
        shift_d=None,
        dist_tau=None,
        device=torch.device('cpu'),
):
    """
    创建一组（一共 O 个）特殊设计的 2D 卷积滤波器（或称为“卷积核”）,每个滤波器是一个条带状的线性滤波器
    tsize: kernal size，决定了扩散的范围,即向周围扩散多少个像素
    resolution: resolution=1即以像素为单位，控制坐标尺度
    sigomid_slope: 控制着sigmoid函数在原地附近的衰减速率
    gaussian_std: 控制着每个滤波器的条带宽度
    shift_d: 控制sigmoid函数中心距离原点坐标的距离,一般不用动
    dist_tau: dist_tau的值是在shift_d值的基础上进行加减，加意味着滤波器的条带向内延申，
        即在滤波器在将概率分布向某个方向，x方向上扩散时，也一定程度上向反方向-x方向扩散；减则意味着只向着x方向扩散
    滤波器组的威力在于可以和位移方向搭配使用，可以慢慢筛选出位移方向
    """
    # make the coord grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=device),
        torch.arange(-(tsize - 1) / 2, (tsize + 1) / 2, 1, device=device),
    )
    # add units,sacle the pix_coord to the coord that you want
    grid_x = grid_x * resolution  # 0.1m
    grid_y = grid_y * resolution  # 0.1m

    # mk orientation for filters
    orns = (torch.arange(0, O, dtype=torch.float32, device=device) / O*2*torch.pi)  # (O,)
    orns = orns + ego if ego is not None else orns
    cos_theta = torch.cos(orns)
    sin_theta = torch.sin(orns)

    shift_d = tsize//2 if shift_d is None else shift_d #tsize//2-x=在tsize//2基础上将sigmoid函数向内移动x个单位
    # 1. 计算每个位移/扩展方向上的sigmoid函数的中心位置,方向x上的sigmoid中心位置由原点坐标向着方向x位移shift_d个单位得到
    #    cos_theta 和 sin_theta 的 shape 为 (O,)，通过 reshape 适配广播机制
    center_x = shift_d * cos_theta.reshape(-1, 1, 1)
    center_y = shift_d * sin_theta.reshape(-1, 1, 1)

    dist_tau = shift_d if dist_tau is None else dist_tau
    # 2. 分量 a): 距离约束 (Sigmoid)，现在计算网格点到 *偏移后中心* 的距离
    #    grid_x/grid_y 的 shape 为 (tsize, tsize)，通过 unsqueeze 适配广播机制
    dist_to_sigmoid_center = torch.sqrt((grid_x.unsqueeze(0) - center_x) ** 2 + (grid_y.unsqueeze(0) - center_y) ** 2) #计算每个grid坐标到sigmoid函数中心的距离
    sigmoid_part = 1 / (1 + torch.exp(sigomid_slope * -(dist_tau-dist_to_sigmoid_center) )) #对于到sigmoid中心距离超过dist_tau的网格坐标，对其进行权重衰减
    # 分量 b): 方向约束 (垂直距离的高斯衰减)
    # 计算点 (grid_x, grid_y) 到方向 theta 所在直线的垂直距离的平方
    # d_perp = |-x*sin + y*cos|
    gaussian_std = resolution*1.5 if gaussian_std is None else gaussian_std
    d_perp_sq = (-grid_x.unsqueeze(0) * sin_theta.reshape(-1, 1, 1) + grid_y.unsqueeze(0) * cos_theta.reshape(-1, 1, 1)) ** 2
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


class GaussianFilter(object):
    """最简单的滤波版本,不关心方向，使用高斯扩散模拟位移；缺点是概率分布峰值点会慢慢滞后于真值点"""
    def __init__(self,
                 filter=None,
                 query_feature=None,
                 gallery_feature=None,
                 query_rc=None,
                 gallery_hw=None,
                 ):
        if filter is None:
            filter_size = 11
            diffusion_L = 5.0  # 假设我们想让概率扩散到半径为5个单位的范围
            pixel_resolution = 1.0  # 每个像素代表1个单位
            gs_fitler = get_gaussian_filter(
                tsize=filter_size,
                L=diffusion_L,
                resolution=pixel_resolution
            )
            self.filter = gs_fitler
        else:
            self.filter = filter
        self.eucdist_computer_feat = LpDistance(normalize_embeddings=True)
        self.query_feature = query_feature
        self.gallery_feature = gallery_feature
        self.query_rc = query_rc
        self.gallery_hw = gallery_hw

    def forward(self,):
        for i, q_feat in enumerate(query_feature):
            # 预测：
            if i == 0:
                bel_ti_ = torch.ones(*gallery_hw, device=device)
            else:
                bel_ti_ = bel_ti
            bel_ti_transed = F.conv2d(bel_ti_.unsqueeze(0).unsqueeze(0),
                                      weight=self.gs_fitler.unsqueeze(0).unsqueeze(0),
                                      padding="same",
                                      ).squeeze()

            # 当前帧的观测结果
            q_feat = self.query_feature[i]
            feat_dist = self.eucdist_computer_feat(gallery_feature, q_feat.unsqueeze(0)).squeeze()
            dist2prob_scale = 2.0
            xy_distb = torch.exp(-dist2prob_scale * feat_dist).reshape(*gallery_hw)
            xy_distb_normed = xy_distb / xy_distb.sum()
            prior = xy_distb_normed

            # debug for vis:
            from vis_featmap import add_gt2maps
            gt_rc = self.query_rc[i] * np.max(self.gallery_hw)
            # vis_single_featmap(xy_distb_normed.reshape(*gallery_hw), return_handles=True)
            # fig,ax = vis_single_featmap(bel_ti_.reshape(*gallery_hw),return_handles=True)
            # add_gt2map(fig, ax,gt_rc)
            fig, axes = vis_multi_featmap(
                torch.stack([xy_distb_normed, bel_ti_, bel_ti_transed / bel_ti_transed.sum()]),
                return_handles=True,
                use_global_scale=False)
            add_gt2maps(fig, axes, np.repeat(gt_rc[None, ...], 3, axis=0))

            # 基于观测的更新：
            if i == 0:
                bel_ti = prior
            else:
                bel_ti = (prior * bel_ti_transed) / (prior * bel_ti_transed).sum()
                # 或者可以考虑平滑项，如何组合当前观测piror和历史信息bel_ti_：
                # beta = 0.1
                # bel_ti = beta*(bel_ti_*prior) + (1-beta)*prior


class LineFilters(object):
    """使用线性滤波器组进行滤波，待调参数是滤波器核大小(tsize)，条带宽度，方向细分数量;
    线性滤波器比gs滤波器强一些，但同样会有一个致命问题，即当迭代次数多了以后，会出现较大的偏差->到底是滞后原因or历史观测不准的原因？
    猜测是历史观测不准导致，因为融合过程依赖乘法操作，一次观测失误(观测的概率峰值远离正确位置)可能会带偏之前所有累计的正确概率分布
    """
    def __init__(self,
                 filter=None,
                 query_feature=None,
                 gallery_feature=None,
                 query_rc=None,
                 gallery_hw=None,
                 device = None,
                 ):

        self.tsize = 13
        self.O = 72
        if filter is None:
            line_filters = get_trans_filters_line(
                tsize=self.tsize,
                O=self.O,
                gaussian_std=1.3,
                sigomid_slope=10,
            )
            self.line_filters = line_filters.to(device)
        else:
            self.line_filters = filter
        self.eucdist_computer_feat = LpDistance(normalize_embeddings=True)
        self.query_feature = query_feature
        self.gallery_feature = gallery_feature
        self.query_rc = query_rc
        self.gallery_hw = gallery_hw
        self.device = device

    def forward(self,):
        O = self.O
        query_feature = self.query_feature
        gallery_feature = self.gallery_feature
        gallery_hw = self.gallery_hw
        query_rc = self.query_rc
        trans_filters = self.line_filters
        device = self.device
        eucdist_computer_feat = self.eucdist_computer_feat

        for i,q_feat in enumerate(query_feature):

            # 预测：
            if i == 0:
                bel_ti_ = torch.ones([O,*gallery_hw], device=device)
            else:
                bel_ti_ = bel_ti.reshape([-1,*gallery_hw])


            bel_ti_transed = F.conv2d(
                bel_ti_.unsqueeze(0),
                weight=trans_filters.unsqueeze(1).flip([-2, -1]), #使用卷积进行概率扩散，而非互相关，对滤波器进行翻转
                bias=None,
                groups=trans_filters.shape[0],
                padding="same",
            ).squeeze()

            #单次观测
            q_feat = query_feature[i]
            feat_dist = eucdist_computer_feat(gallery_feature, q_feat.unsqueeze(0)).squeeze()
            dist2prob_scale = 2.0
            xy_distb = torch.exp(-dist2prob_scale * feat_dist).reshape(*gallery_hw)
            xy_distb_normed = xy_distb / xy_distb.sum()

            #融合单次观测与历史信息
            prior = xy_distb_normed
            if i==0:
                bel_ti = prior.unsqueeze(0).repeat(O,1,1)
            else:
                bel_ti = prior.unsqueeze(0) * bel_ti_transed
                bel_ti = bel_ti/bel_ti.sum()*O

                #debug for vis:
                prob_xy, orientations = torch.max(bel_ti, dim=0)
                pred_y, pred_x = torch.where(prob_xy == prob_xy.max())
                gt_rc = (query_rc[i]*np.max(prob_xy.shape))
                # orn = orientations[pred_y, pred_x]*(360/O)
                # pred_xyr = torch.tensor([pred_x, pred_y, orn]).cpu().numpy()
                # singview
                # fig,ax = vis_single_featmap(prob_xy,return_handles=True)
                # add_pts2map(fig,ax,np.stack([gt_rc,np.array([pred_y.cpu().numpy(),pred_x.cpu().numpy()]).squeeze()]),labels=['gt','pred'],title=i)
                # multi-view
                fig, axes = vis_multi_featmap(
                    torch.stack([prior,prob_xy]),
                    return_handles=True,
                    use_global_scale=False)
                from vis_featmap import add_gt2maps
                add_gt2maps(fig, axes, np.repeat(gt_rc[None, ...], 2, axis=0),title=i)


import torch.nn.functional as F
from vis_featmap import vis_single_featmap, vis_multi_featmap, add_pts2map
#####################################################
# #################
if __name__ == "__main__":
    path2res_mat = '/home/data/zwk/pyproj_DUAV_salad_6.4/exps/exp24/epoch003_overlap0.75_radius32m.mat'
    with_sg = False
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
    gallery_hw = result['gallery_hw'][0]

    gallery_feature = gallery_feature.to(device)
    query_feature = query_feature.to(device)

    if with_sg:
        gallery_sg = torch.FloatTensor(result['gallery_sg'])
        query_sg = torch.FloatTensor(result['query_sg'])
        rotdeg_fm_north_anticlock = result['rotdeg_fm_north_anticlock'][0]
        rotdeg_fm_north_anticlock_positive = rotdeg_fm_north_anticlock[-query_feature.shape[0]:]
        relrot_normed = norm_rot(rotdeg_fm_north_anticlock_positive)
        gallery_sg = gallery_sg.to(device)
        query_sg = query_sg.to(device)

    linefilters = LineFilters(
        query_feature=query_feature,
        gallery_feature=gallery_feature,
        query_rc=query_rc,
        gallery_hw=gallery_hw,
        device=device,
    )
    linefilters.forward()

    #transit by filters,debug the trans_fitler here
    tsize=15
    O=72

    trans_filters = get_trans_filters_halfline(
        tsize=tsize,
        gaussian_std=1.5,
        sigomid_slope=2.5,
        shift_d=tsize//2,
        dist_tau=tsize//2+2,
        O=O,
    )
    # vis_multi_featmap(trans_filters[:4])
    # filter = line_filters.sum(dim=0)
    # vis_3d_surface(line_filters[0])

    trans_filters = get_trans_filters_line(
        tsize=tsize,
        O=O,
        gaussian_std=1.3,
        sigomid_slope=10,
    )
    vis_multi_featmap(trans_filters[:4])

    trans_filters = trans_filters.to(device)
    eucdist_computer_feat = LpDistance(normalize_embeddings=True)
    for i,q_feat in enumerate(query_feature):

        # 预测：
        if i == 0:
            bel_ti_ = torch.ones([O,*gallery_hw], device=device)
        else:
            bel_ti_ = bel_ti.reshape([-1,*gallery_hw])

            #debug for vis:
            prob_xy, orientations = torch.max(bel_ti_, dim=0)
            pred_y, pred_x = torch.where(prob_xy == prob_xy.max())
            orn = orientations[pred_y, pred_x]*(360/O)
            pred_xyr = torch.tensor([pred_x, pred_y, orn]).cpu().numpy()
            gtrc = (query_rc[i]*np.max(prob_xy.shape))
            fig,ax=vis_single_featmap(prob_xy,return_handles=True)
            add_pts2map(fig,ax,np.stack([gtrc,np.array([pred_y.cpu().numpy(),pred_x.cpu().numpy()]).squeeze()]),labels=['gt','pred'],title=i)
            # gtr = relrot_normed[i]
            # gt_xyr = np.array([gtxy[0],gtxy[1], gtr])
            # p2save = os.path.join('exps/debug_vis',f'{i}.png')
            # vis_single_featmap_with_direct(prob_dist,p2save=p2save,title=f'{i}th',xyrot1=pred_xyr,xyrot2=gt_xyr,camp='coolwarm')
            # vis_single_featmap_with_direct(prob_dist,p2save=None,title=f'{i}th',xyrot1=pred_xyr,xyrot2=gt_xyr,camp='coolwarm')

        bel_ti_transed = F.conv2d(
            bel_ti_.unsqueeze(0),
            weight=trans_filters.unsqueeze(1).flip([-2, -1]), #使用卷积进行概率扩散，而非互相关，对滤波器进行翻转
            bias=None,
            groups=trans_filters.shape[0],
            padding="same",
        ).squeeze()

        #单次观测
        q_feat = query_feature[i]
        feat_dist = eucdist_computer_feat(gallery_feature, q_feat.unsqueeze(0)).squeeze()
        dist2prob_scale = 2.0
        xy_distb = torch.exp(-dist2prob_scale * feat_dist).reshape(*gallery_hw)
        xy_distb_normed = xy_distb / xy_distb.sum()

        #融合单次观测与历史信息
        prior = xy_distb_normed
        if i==0:
            bel_ti = prior.unsqueeze(0).repeat(O,1,1)
        else:
            bel_ti = prior.unsqueeze(0) * bel_ti_transed
            bel_ti = bel_ti/bel_ti.sum()*O




