# todo:
    tools/* -> trainer_depends/*
    清理无用函数&工具&历史版本

# 进行中：
    stage3_bpf_proxy_linearProjector_wANCE_evotorch.py:尝试sms_loss
    stage3_bpf_proxy_linearProjector_wANCE_evotorch.py:作为 stage3_bpf_proxy_linearProjector_wANCE.py 的重大升级入口，后续 CMA-ES / 共享GPU评估 / EvoTorch 相关修改优先落在此文件
    trainers/util_stage3_multi_start_CMAES_by_evotorch.py: EvoTorch 专用多起点 CMA-ES 工具文件；和旧 cmaes 版 util 分离，避免后续逻辑混杂

# traners/下的可用版本：
## stage1可用版：
    trainers/stage1_visual_encoder_w_ANCE.py

## stage2的可用版： 
    trainers/stage2_grid_hashfit.py

## stage3的可用版：
    trainers/stage3_bpf_proxy_linearProjector_wANCE.py,至少测试是完整的，可以使用cames库的cam-es算法
    trainers/stage3_bpf_proxy_linearProjector_wANCE_evotorch.py, stage3_bpf_proxy_linearProjector_wANCE.py 的升级版本；新算法线优先在这里迭代，尤其是 CMA-ES / 共享GPU评估 / EvoTorch 方向
    stage3_bpf_proxy_linearProjector.py = 线性投影，CL版
    stage3_project_integrateRot_classify.py = 线性投影，积分推断版
    stage3_project_integrateRot_*** =待完善版
