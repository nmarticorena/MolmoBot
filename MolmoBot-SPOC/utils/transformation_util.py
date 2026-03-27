import torch
import torchvision.transforms
from torchvision.datasets import ImageNet
import random
from torchvision.transforms import Compose, Normalize

from molmobot_spoc.utils.image_warping_utils import (
    warp_image_gpu,
    calc_camera_intrinsics,
    get_randomized_distortion_parameters,
)
from molmobot_spoc.utils.constants.camera_constants import (
    GOPRO_VERTICAL_FOV,
    GOPRO_CAMERA_HEIGHT,
    GOPRO_CAMERA_WIDTH,
)


def get_full_transformation_list(size, version="v2"):
    if version == "v2":
        return [
            torchvision.transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05
            ),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
            # torchvision.transforms.RandomResizedCrop(
            #     size,
            #     scale=(0.9, 1),
            # ),
            torchvision.transforms.RandomPosterize(bits=7, p=0.2),
            torchvision.transforms.RandomPosterize(bits=6, p=0.2),
            torchvision.transforms.RandomPosterize(bits=5, p=0.2),
            torchvision.transforms.RandomPosterize(bits=4, p=0.2),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ]
    elif version == "v1":
        return [
            torchvision.transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
            ),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2)),
            # torchvision.transforms.RandomResizedCrop(
            #     size,
            #     scale=(0.9, 1),
            # ),
            torchvision.transforms.RandomPosterize(bits=7, p=0.3),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=2),
            torchvision.transforms.RandomGrayscale(0.2),
        ]
    else:
        raise NotImplementedError(
            f"data augmentation versions supported are v1 and v2, got {name}"
        )


def get_transformation(size=(224, 384)):
    list_of_transformations = get_full_transformation_list(size)
    return Compose(list_of_transformations)


def sample_a_specific_transform(transformation_list, size=(224, 384)):
    specific_transformation = []
    for transform in transformation_list.transforms:

        def sample_value_in_range(list_of_range):
            assert len(list_of_range) == 2
            random_value = random.uniform(list_of_range[0], list_of_range[1])
            return (random_value, random_value)

        def sample_singular_value(prob):
            return int(random.random() < prob)

        if type(transform) == torchvision.transforms.ColorJitter:
            sampled_brightness = sample_value_in_range(transform.brightness)
            sampled_saturation = sample_value_in_range(transform.saturation)
            sampled_hue = sample_value_in_range(transform.hue)
            sampled_contrast = sample_value_in_range(transform.contrast)

            specific_transformation.append(
                torchvision.transforms.ColorJitter(
                    brightness=sampled_brightness,
                    contrast=sampled_contrast,
                    saturation=sampled_saturation,
                    hue=sampled_hue,
                )
            )
        elif type(transform) == torchvision.transforms.GaussianBlur:
            sampled_sigma = sample_value_in_range(transform.sigma)

            specific_transformation.append(
                torchvision.transforms.GaussianBlur(
                    kernel_size=transform.kernel_size, sigma=sampled_sigma
                )
            )
        elif type(transform) == torchvision.transforms.RandomResizedCrop:
            sampled_scale = sample_value_in_range(transform.scale)

            specific_transformation.append(
                torchvision.transforms.RandomResizedCrop(
                    size,
                    scale=sampled_scale,
                )
            )

        elif type(transform) == torchvision.transforms.RandomPosterize:
            sampled_p = sample_singular_value(transform.p)

            specific_transformation.append(
                torchvision.transforms.RandomPosterize(bits=7, p=sampled_p)
            )
        elif type(transform) == torchvision.transforms.RandomAdjustSharpness:
            sampled_p = sample_singular_value(transform.p)

            specific_transformation.append(
                torchvision.transforms.RandomAdjustSharpness(
                    sharpness_factor=2, p=sampled_p
                )
            )
        elif type(transform) == torchvision.transforms.RandomGrayscale:
            sampled_p = sample_singular_value(transform.p)

            specific_transformation.append(
                torchvision.transforms.RandomGrayscale(p=sampled_p)
            )
        elif type(transform) in [torchvision.transforms.Lambda or Normalize]:
            specific_transformation.append(transform)
        else:
            raise NotImplementedError

    return Compose(specific_transformation)


class ApplyFisheyeWarping(torch.nn.Module):
    def __init__(self, K=None, perturbation_magnitude=0.00001):
        super().__init__()
        self.H = GOPRO_CAMERA_HEIGHT
        self.W = GOPRO_CAMERA_WIDTH
        self.perturbation_magnitude = (
            perturbation_magnitude  # Controls the magnitude of random perturbations
        )

        # Initialize and register K for standard values
        if K is None:
            K = calc_camera_intrinsics(GOPRO_VERTICAL_FOV, self.H, self.W)
        self.register_buffer("K", torch.tensor(K, dtype=torch.float32))

        # Register buffers for precomputed values
        self.register_buffer("x_normalized", None)
        self.register_buffer("y_normalized", None)
        self.register_buffer("r", None)

        self._precompute_values()

    def _precompute_values(self):
        device = self.K.device
        y, x = torch.meshgrid(
            torch.arange(self.H, device=device).float(),
            torch.arange(self.W, device=device).float(),
        )
        self.x_normalized = (x - self.K[0, 2]) / self.K[0, 0]
        self.y_normalized = (y - self.K[1, 2]) / self.K[1, 1]
        self.r = torch.sqrt(self.x_normalized**2 + self.y_normalized**2)

    def _get_zero_mean_perturbation_parameters(self):
        # Create small random perturbations centered around zero
        magnitude = self.perturbation_magnitude
        return {
            "k1": random.gauss(0, magnitude),
            "k2": random.gauss(0, magnitude),
            "k3": random.gauss(0, magnitude),
            "k4": random.gauss(0, magnitude),
        }

    def forward(self, img):
        # Ensure all tensors are on the same device as the input image
        device = img.device
        K = self.K.to(device)

        # Generate small zero-mean random perturbations
        distortion_parameters = self._get_zero_mean_perturbation_parameters()

        x_normalized = self.x_normalized.to(device)
        y_normalized = self.y_normalized.to(device)
        r = self.r.to(device)

        return warp_image_gpu(
            image=img,
            K=K,
            distortion_parameters=distortion_parameters,
            x_normalized=x_normalized,
            y_normalized=y_normalized,
            r=r,
            crop_percent=0.0,
        )


class ApplyFullFisheyeWarping(torch.nn.Module):
    def __init__(self, H=None, W=None, K=None):
        super().__init__()
        self.H = H if H is not None else GOPRO_CAMERA_HEIGHT
        self.W = W if W is not None else GOPRO_CAMERA_WIDTH

        # Initialize and register K for standard values
        if K is None:
            K = calc_camera_intrinsics(GOPRO_VERTICAL_FOV, self.H, self.W)
        self.register_buffer("K", torch.tensor(K, dtype=torch.float32))

        # Register buffers for precomputed values
        self.register_buffer("x_normalized", None)
        self.register_buffer("y_normalized", None)
        self.register_buffer("r", None)

        self._precompute_values()

    def _precompute_values(self):
        device = self.K.device
        y, x = torch.meshgrid(
            torch.arange(self.H, device=device).float(),
            torch.arange(self.W, device=device).float(),
        )
        self.x_normalized = (x - self.K[0, 2]) / self.K[0, 0]
        self.y_normalized = (y - self.K[1, 2]) / self.K[1, 1]
        self.r = torch.sqrt(self.x_normalized**2 + self.y_normalized**2)

    def forward(self, img):
        # Ensure all tensors are on the same device as the input image
        device = img.device
        K = self.K.to(device)

        # Randomize distortion parameters in the forward pass
        distortion_parameters = get_randomized_distortion_parameters()

        x_normalized = self.x_normalized.to(device)
        y_normalized = self.y_normalized.to(device)
        r = self.r.to(device)

        return warp_image_gpu(
            image=img,
            K=K,
            distortion_parameters=distortion_parameters,
            x_normalized=x_normalized,
            y_normalized=y_normalized,
            r=r,
        )
