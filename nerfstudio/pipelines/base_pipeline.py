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
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import typing
from abc import abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, Type, Union, cast

import torch
import torch.distributed as dist
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from torch import nn
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    FlexibleDataManager,
    FlexibleDataManagerConfig,
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.pipelines.adaptive_feedback import submit_adaptive_feedback
from nerfstudio.utils import profiler
from nerfstudio.utils.images import BasicImages


def module_wrapper(ddp_or_model: Union[DDP, Model]) -> Model:
    """
    If DDP, then return the .module. Otherwise, return the model.
    """
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return ddp_or_model


class Pipeline(nn.Module):
    """The intent of this class is to provide a higher level interface for the Model
    that will be easy to use for our Trainer class.
    """

    # pylint: disable=abstract-method

    datamanager: DataManager
    _model: Model

    @property
    def model(self):
        """Returns the unwrapped model if in ddp"""
        return module_wrapper(self._model)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """Get training loss dict."""
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        submit_adaptive_feedback(self.datamanager, batch, model_outputs, loss_dict)

        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """Get evaluation loss dict."""
        self.eval()
        if self.world_size > 1:
            assert self.datamanager.eval_sampler is not None
            self.datamanager.eval_sampler.set_epoch(step)
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle, batch)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @abstractmethod
    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """Abstract."""

    @abstractmethod
    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Abstract."""

    def load_pipeline(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path"""

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline."""


@dataclass
class VanillaPipelineConfig(cfg.InstantiateConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VanillaPipeline)
    """target class to instantiate"""
    datamanager: VanillaDataManagerConfig = VanillaDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()
    """specifies the model config"""


class VanillaPipeline(Pipeline):
    """Pipeline for vanilla setup."""

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.test_mode = test_mode
        self.datamanager: VanillaDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        self.datamanager.to(device)
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            world_size=world_size,
            local_rank=local_rank,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.model.device

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """Get training loss dict."""
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
        if camera_opt_param_group in self.datamanager.get_param_groups():
            metrics_dict["camera_opt_translation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
            )
            metrics_dict["camera_opt_rotation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
            )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        submit_adaptive_feedback(self.datamanager, batch, model_outputs, loss_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Blank forward method"""
        raise NotImplementedError

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """Get evaluation loss dict."""
        self.eval()
        ray_bundle, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """Eval images."""
        self.eval()
        torch.cuda.empty_cache()
        image_idx, camera_ray_bundle, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        assert "image_idx" not in metrics_dict
        metrics_dict["image_idx"] = image_idx
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = len(camera_ray_bundle)
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Average over all eval images."""
        self.eval()
        metrics_dict_list = []
        images_dict_list = []
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                isbasicimages = False
                if isinstance(batch["image"], BasicImages):
                    isbasicimages = True
                    batch["image"] = batch["image"].images[0]
                    camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1],))
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                images_dict_list.append(images_dict)
                progress.advance(task)
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([m[key] for m in metrics_dict_list]))
            )
        self.train()
        return metrics_dict, images_dict_list

    @profiler.time_function
    def get_visibility_mask(
        self,
        coarse_grid_resolution: int = 512,
        valid_points_thres: float = 0.005,
        sub_sample_factor: int = 8,
    ):
        """Compute visibility mask."""
        self.eval()

        coarse_mask = torch.ones(
            (1, 1, coarse_grid_resolution, coarse_grid_resolution, coarse_grid_resolution), requires_grad=True
        ).to(self.device)
        coarse_mask.retain_grad()

        num_images = len(self.datamanager.fixed_indices_train_dataloader)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_train_dataloader:
                isbasicimages = False
                if isinstance(batch["image"], BasicImages):
                    isbasicimages = True
                    batch["image"] = batch["image"].images[0]
                    camera_ray_bundle = camera_ray_bundle.reshape((*batch["image"].shape[:-1],))
                camera_ray_bundle = camera_ray_bundle[::sub_sample_factor, ::sub_sample_factor]
                height, width = camera_ray_bundle.shape
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                ray_points = outputs["ray_points"].reshape(height, width, -1, 3)
                weights = outputs["weights"]

                valid_points = ray_points.reshape(-1, 3)[weights.reshape(-1) > valid_points_thres]
                valid_points = valid_points * 0.5  # normalize from [-2, 2] to [-1, 1]
                with torch.enable_grad():
                    out = torch.nn.functional.grid_sample(coarse_mask, valid_points[None, None, None])
                    out.sum().backward()
                progress.advance(task)

        coarse_mask = (coarse_mask.grad > 0.0001).float()
        self.train()
        return coarse_mask

    def load_pipeline(self, loaded_state: Dict[str, Any]) -> None:
        """Load the checkpoint from the given path"""
        state = {key.replace("module.", ""): value for key, value in loaded_state.items()}
        if self.test_mode == "inference":
            state.pop("datamanager.train_camera_optimizer.pose_adjustment", None)
            state.pop("datamanager.train_ray_generator.image_coords", None)
            state.pop("datamanager.train_ray_generator.pose_optimizer.pose_adjustment", None)
            state.pop("datamanager.eval_ray_generator.image_coords", None)
            state.pop("datamanager.eval_ray_generator.pose_optimizer.pose_adjustment", None)
        self.load_state_dict(state)  # type: ignore

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        """Returns callbacks from datamanager + model."""
        datamanager_callbacks = self.datamanager.get_training_callbacks(training_callback_attributes)
        model_callbacks = self.model.get_training_callbacks(training_callback_attributes)
        callbacks = datamanager_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Return param groups."""
        datamanager_params = self.datamanager.get_param_groups()
        model_params = self.model.get_param_groups()
        return {**datamanager_params, **model_params}


@dataclass
class FlexibleInputPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: FlexibleInputPipeline)
    """target class to instantiate"""
    datamanager: FlexibleDataManagerConfig = FlexibleDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = ModelConfig()


class FlexibleInputPipeline(VanillaPipeline):
    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """Get training loss dict with additional inputs."""
        ray_bundle, batch, additional_input = self.datamanager.next_train(step)
        model_outputs = self.model.get_outputs_flexible(ray_bundle, additional_input)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
        if camera_opt_param_group in self.datamanager.get_param_groups():
            metrics_dict["camera_opt_translation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
            )
            metrics_dict["camera_opt_rotation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
            )

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        submit_adaptive_feedback(self.datamanager, batch, model_outputs, loss_dict)

        return model_outputs, loss_dict, metrics_dict
