# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Datamanager.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import tyro
from rich.progress import Console
from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.friends_dataparser import FriendsDataParserConfig
from nerfstudio.data.dataparsers.heritage_dataparser import HeritageDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.mipnerf360_dataparser import Mipnerf360DataParserConfig
from nerfstudio.data.dataparsers.monosdf_dataparser import MonoSDFDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
# from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import PhototourismDataParserConfig
from nerfstudio.data.dataparsers.record3d_dataparser import Record3DDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.datasets.base_dataset import GeneralizedDataset, InputDataset
# Adaptive sampler
from nerfstudio.data.pixel_samplers import (
    EquirectangularPixelSampler,
    PixelSampler,
    AdaptivePixelSampler,
)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.images import BasicImages
from nerfstudio.utils.misc import IterableWrapper

CONSOLE = Console(width=120)

AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[
    # Omit prefixes of flags in subcommands.
    tyro.extras.subcommand_type_from_defaults(
        {
            "nerfstudio-data": NerfstudioDataParserConfig(),
            "mipnerf360-data": Mipnerf360DataParserConfig(),
            "blender-data": BlenderDataParserConfig(),
            "friends-data": FriendsDataParserConfig(),
            "instant-ngp-data": InstantNGPDataParserConfig(),
            # "nuscenes-data": NuScenesDataParserConfig(),
            "record3d-data": Record3DDataParserConfig(),
            "dnerf-data": DNeRFDataParserConfig(),
            "phototourism-data": PhototourismDataParserConfig(),
            "monosdf-data": MonoSDFDataParserConfig(),
            "sdfstudio-data": SDFStudioDataParserConfig(),
            "heritage-data": HeritageDataParserConfig(),
        },
        prefix_names=False,  # Omit prefixes in subcommands themselves.
    )
]
# Union over possible dataparser types, annotated with metadata for tyro.
# This is the same as the vanilla union, but results in shorter subcommand names.


class DataManager(nn.Module):
    """Generic data manager's abstract class

    Usage:
        To get data, use the next_train and next_eval functions.
        This data manager's next_train and next_eval methods will return:
        1) RayBundle with rays to sample
        2) Pixel batch dict (GT pixels, masks, etc.)
    """

    train_dataset: Optional[Dataset] = None
    eval_dataset: Optional[Dataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None

    def __init__(self) -> None:
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_dataset and self.test_mode != "inference":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
            self.setup_eval()

    def forward(self) -> None:
        raise NotImplementedError

    def iter_train(self) -> None:
        self.train_count = 0

    def iter_eval(self) -> None:
        self.eval_count = 0

    def get_train_iterable(self, length: int = -1) -> IterableWrapper:
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length: int = -1) -> IterableWrapper:
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def setup_eval(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def next_train(self, step: int) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple:
        raise NotImplementedError

    def get_training_callbacks(  # pylint:disable=no-self-use
        self,
        training_callback_attributes: TrainingCallbackAttributes,  # pylint: disable=unused-argument
    ) -> List[TrainingCallback]:
        return []

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        return {}


@dataclass
class VanillaDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of
    keeping the train/eval dataparsers; After instantiation, data manager holds both
    train/eval datasets and is in charge of returning unpacked train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: VanillaDataManager)

    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()

    train_num_rays_per_batch: int = 1024
    train_num_images_to_sample_from: int = -1
    train_num_times_to_repeat_images: int = -1

    eval_num_rays_per_batch: int = 1024
    eval_num_images_to_sample_from: int = -1
    eval_num_times_to_repeat_images: int = -1
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)

    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    collate_fn = staticmethod(nerfstudio_collate)

    camera_res_scale_factor: float = 1.0

    # Adaptive sampler configs
    sampler_type: Literal["default", "adaptive"] = "adaptive"
    adaptive_edge_type: Literal["sobel", "canny"] = "sobel"
    adaptive_edge_weight: float = 1.3
    adaptive_uniform_mix: float = 0.6
    adaptive_ema_decay: float = 0.9
    adaptive_hardness_enabled: bool = True

    # Hierarchical allocation; the paper samples uniformly within each cell.
    adaptive_block_size: int = 8  # 0/1: pixel-level; >1: cell side length
    adaptive_blockwise_exact: bool = False


class VanillaDataManager(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation."""

    config: VanillaDataManagerConfig
    train_dataset: InputDataset
    eval_dataset: InputDataset

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"

        self.dataparser = self.config.dataparser.setup()

        # mark building target (train/eval) so _get_pixel_sampler can choose
        self._building_eval: bool = False

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()

        super().__init__()

    def create_train_dataset(self) -> InputDataset:
        """Sets up the data loaders for training"""
        return GeneralizedDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split="train"),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> InputDataset:
        """Sets up the data loaders for evaluation"""
        return GeneralizedDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(  # pylint: disable=no-self-use
        self,
        dataset: InputDataset,
        *args: Any,
        **kwargs: Any,
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        is_equirectangular = dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        if is_equirectangular.any():
            CONSOLE.print(
                "[bold yellow]Warning: Some cameras are equirectangular, "
                "but using default pixel sampler."
            )

        # choose adaptive only for training
        use_adaptive = (not self._building_eval) and (self.config.sampler_type == "adaptive")

        # number of rays passed in
        num_rays = (
            args[0] if len(args) > 0 else kwargs.get("num_rays_per_batch", self.config.train_num_rays_per_batch)
        )
        keep_full_image = kwargs.get("keep_full_image", False)

        if use_adaptive:
            return AdaptivePixelSampler(
                num_rays_per_batch=num_rays,
                keep_full_image=keep_full_image,
                ema_decay=self.config.adaptive_ema_decay,
                hardness_enabled=self.config.adaptive_hardness_enabled,
                uniform_mix=self.config.adaptive_uniform_mix,
                edge_weight=self.config.adaptive_edge_weight,
                edge_type=self.config.adaptive_edge_type,
                device=self.device,
                # blockwise options
                block_size=self.config.adaptive_block_size,
                blockwise_exact=self.config.adaptive_blockwise_exact,
            )

        # default uniform sampler
        return PixelSampler(num_rays_per_batch=num_rays, keep_full_image=keep_full_image)

    def setup_train(self) -> None:
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")

        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

        # mark: building train sampler (enable adaptive)
        self._building_eval = False
        self.train_pixel_sampler = self._get_pixel_sampler(
            self.train_dataset, self.config.train_num_rays_per_batch
        )

        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size,
            device=self.device,
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

        # for loading full images
        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )

    def setup_eval(self) -> None:
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")

        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 2,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)

        # mark: building eval sampler (force default/random)
        self._building_eval = True
        self.eval_pixel_sampler = self._get_pixel_sampler(
            self.eval_dataset, self.config.eval_num_rays_per_batch
        )
        self._building_eval = False  # reset

        self.eval_ray_generator = RayGenerator(
            self.eval_dataset.cameras.to(self.device),
            self.train_camera_optimizer,  # should be shared between train and eval.
        )

        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )

        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            image_indices=self.config.eval_image_indices,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, batch in self.eval_dataloader:
            assert camera_ray_bundle.camera_indices is not None
            if isinstance(batch["image"], BasicImages):
                # generalized dataset → get tensor
                batch["image"] = batch["image"].images[0]
                camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1], 1))
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])
            return image_idx, camera_ray_bundle, batch

        raise ValueError("No more eval images")

    # convenient hook to feed per-ray losses back to sampler
    @torch.no_grad()
    def update_from_per_ray_loss(self, batch: Dict, loss_dict: Dict, key: str = "per_ray_l2") -> bool:
        """Feed per-ray loss back to adaptive sampler for EMA 'hardness' update."""
        if not hasattr(self, "train_pixel_sampler"):
            return False
        sampler = self.train_pixel_sampler
        if not hasattr(sampler, "update"):
            return False
        if key not in loss_dict or "indices" not in batch:
            return False
        sampler.update(batch["indices"], loss_dict[key])
        return True

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager."""
        param_groups: Dict[str, List[Parameter]] = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups


@dataclass
class FlexibleDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for data manager instantiation; DataManager is in charge of
    keeping the train/eval dataparsers; After instantiation, data manager holds both
    train/eval datasets and is in charge of returning unpacked train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: FlexibleDataManager)
    train_num_images_to_sample_from: int = 1


class FlexibleDataManager(VanillaDataManager):
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        additional_output: Dict[str, Any] = {}
        if "src_imgs" in image_batch.keys():
            ray_indices = ray_indices.to(image_batch["src_idxs"].device)
            assert (ray_indices[:, 0] == image_batch["image_idx"]).all()
            additional_output["uv"] = ray_indices[:, 1:]
            additional_output["src_idxs"] = image_batch["src_idxs"][0]
            additional_output["src_imgs"] = image_batch["src_imgs"][0]
            additional_output["src_cameras"] = self.train_dataset._dataparser_outputs.cameras[
                image_batch["src_idxs"][0]
            ]

        return ray_bundle, batch, additional_output
