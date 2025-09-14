import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def mk_gta_transform(img_size,mean,std,p_rot=1.0):
    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15,
                                                    always_apply=False, p=0.5),
                                      A.OneOf([
                                          A.AdvancedBlur(p=1.0),
                                          A.Sharpen(p=1.0),
                                      ], p=0.3),
                                      A.OneOf([
                                          A.GridDropout(ratio=0.4, p=1.0),
                                          A.CoarseDropout(max_holes=25,
                                                          max_height=int(0.2 * img_size),
                                                          max_width=int(0.2 * img_size),
                                                          min_holes=10,
                                                          min_height=int(0.1 * img_size),
                                                          min_width=int(0.1 * img_size),
                                                          p=1.0),
                                      ], p=0.3),
                                      A.RandomRotate90(p=p_rot),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])
    return train_sat_transforms