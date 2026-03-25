import numpy as np


class MultiSceneDataLoader:
    """
    多场景数据加载器

    支持多种采样策略：
    - round_robin: 轮流采样各场景
    - random: 随机采样场景
    - weighted: 按权重采样场景
    """

    def __init__(self, dataloaders, sampling_strategy='round_robin'):
        self.dataloaders = dataloaders
        self.scene_names = list(dataloaders.keys())
        self.num_scenes = len(self.scene_names)
        self.sampling_strategy = sampling_strategy

        self.total_batches = sum(len(dl) for dl in dataloaders.values())
        self.scene_iters = {}
        self.current_scene_idx = 0
        self.current_iter = 0

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        self.scene_iters = {name: iter(dl) for name, dl in self.dataloaders.items()}
        self.current_scene_idx = 0
        self.current_iter = 0
        return self

    def __next__(self):
        if self.current_iter >= self.total_batches:
            raise StopIteration

        if self.sampling_strategy == 'round_robin':
            scene_name = self.scene_names[self.current_scene_idx]
            self.current_scene_idx = (self.current_scene_idx + 1) % self.num_scenes
        elif self.sampling_strategy == 'random':
            scene_name = np.random.choice(self.scene_names)
        elif self.sampling_strategy == 'weighted':
            weights = [self.dataloaders[name].dataset.weight for name in self.scene_names]
            scene_name = np.random.choice(self.scene_names, p=np.array(weights) / sum(weights))
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

        try:
            batch = next(self.scene_iters[scene_name])
        except StopIteration:
            self.scene_iters[scene_name] = iter(self.dataloaders[scene_name])
            batch = next(self.scene_iters[scene_name])

        batch['scene_name'] = scene_name
        self.current_iter += 1
        return batch
