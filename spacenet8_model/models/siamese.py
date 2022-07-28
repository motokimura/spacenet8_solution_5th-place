import segmentation_models_pytorch as smp
import torch

# isort: off
from spacenet8_model.utils.misc import get_flatten_classes
# isort: on


class SiameseModel(torch.nn.Module):
    def __init__(self, config, **kwargs):
        assert config.Model.n_input_post_images in [1, 2], config.Model.n_input_post_images

        super().__init__()

        # siamese branch
        n_classes = len(get_flatten_classes(config))
        self.branch = smp.create_model(
            config.Model.arch,
            encoder_name=config.Model.encoder,
            in_channels=3,
            classes=n_classes,
            encoder_weights="imagenet",
            **kwargs)

        branch_out_channels = self.branch.segmentation_head[0].in_channels
        self.branch.segmentation_head[0] = torch.nn.Identity()

        # siamese head
        head_in_channels = branch_out_channels * (1 + config.Model.n_input_post_images)
        kernel_size = config.Model.siamese_head_kernel_size
        padding = (kernel_size - 1) // 2
        assert config.Model.n_siamese_head_convs >= 1
        head = [torch.nn.Conv2d(
                    head_in_channels,
                    head_in_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding) for _ in range(config.Model.n_siamese_head_convs - 1)
                ]
        head.append(
            torch.nn.Conv2d(
                head_in_channels,
                n_classes,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        self.head = torch.nn.Sequential(*head)

        self.n_input_post_images = config.Model.n_input_post_images

    def forward(self, image, images_post):
        assert len(images_post) == self.n_input_post_images

        x = [self.branch(image)]
        for i in range(self.n_input_post_images):
            x.append(self.branch(images_post[i]))
        x = torch.cat(x, dim=1)
        return self.head(x)
