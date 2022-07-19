import albumentations as albu


def get_transforms(config, is_train):
    if is_train:
        transforms = [
            albu.RandomCrop(
                width=config.Transform.train_random_crop_size[0],
                height=config.Transform.train_random_crop_size[1],
                always_apply=True),
        ]
    else:
        transforms = [
            albu.PadIfNeeded(
                pad_height_divisor=32,
                pad_width_divisor=32,
                min_height=None,
                min_width=None,
                always_apply=True,
                border_mode=0,  # 0: cv2.BORDER_CONSTANT
                value=0,
                mask_value=0
            ),
        ]
    return albu.Compose(transforms)
