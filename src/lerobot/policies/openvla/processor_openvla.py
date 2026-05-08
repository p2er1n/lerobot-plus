from __future__ import annotations

from typing import Any, ClassVar

import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ActionProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_openvla import OpenVLAConfig


def letterbox_pad_transform(image: Image.Image, padding_fill_value: tuple[int, int, int]) -> Image.Image:
    """Pad a PIL image to a square canvas using the upstream Prismatic letterbox rule."""
    (width, height), max_wh = image.size, max(image.size)
    horizontal_pad = int((max_wh - width) / 2)
    vertical_pad = int((max_wh - height) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    """Local copy of the upstream OpenVLA HF image processor.

    The class is kept out of the LeRobot processor pipeline for now. It exists so the
    adapter can reuse upstream resize/normalize semantics from local code instead of
    importing them from the OpenVLA submodule later on.
    """

    model_input_names: ClassVar[list[str]] = ["pixel_values"]

    def __init__(
        self,
        use_fused_vision_backbone: bool = False,
        image_resize_strategy: str = "letterbox",
        input_sizes: list[tuple[int, int, int]] | None = None,
        interpolations: list[str] | None = None,
        means: list[tuple[float, float, float]] | None = None,
        stds: list[tuple[float, float, float]] | None = None,
        **kwargs: str,
    ) -> None:
        try:
            import timm.data
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PrismaticImageProcessor requires the timm package. "
                "Install timm in the lerobot-plus environment before using OpenVLA local image processing."
            ) from exc

        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_resize_strategy = image_resize_strategy
        self.input_sizes = [(3, 224, 224)] if input_sizes is None else input_sizes
        self.interpolations = interpolations
        self.means = [(0.5, 0.5, 0.5)] if means is None else means
        self.stds = [(0.5, 0.5, 0.5)] if stds is None else stds

        self.tvf_resize_params: list[dict[str, Any]] = []
        self.tvf_crop_params: list[dict[str, Any]] = []
        self.tvf_normalize_params: list[dict[str, Any]] = []
        self.tvf_do_letterbox = False
        self.tvf_letterbox_fill: tuple[int, int, int] | None = None

        for idx in range(len(self.input_sizes)):
            transform = timm.data.create_transform(
                input_size=self.input_sizes[idx],
                interpolation=None if self.interpolations is None else self.interpolations[idx],
                mean=self.means[idx],
                std=self.stds[idx],
                crop_pct=1.0,
                crop_mode="center",
                is_training=False,
            )
            if not (
                isinstance(transform, Compose)
                and len(transform.transforms) == 4
                and isinstance(transform.transforms[0], Resize)
                and isinstance(transform.transforms[1], CenterCrop)
                and isinstance(transform.transforms[2], ToTensor)
                and isinstance(transform.transforms[3], Normalize)
            ):
                raise ValueError(f"Unexpected TIMM image transformation structure: {transform}")

            resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
            self.tvf_resize_params.append(
                {
                    "size": resize_t.size,
                    "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
                    "max_size": None,
                    "antialias": True,
                }
            )
            self.tvf_crop_params.append({"output_size": crop_t.size})
            self.tvf_normalize_params.append(
                {
                    "mean": norm_t.mean.float().numpy().tolist(),
                    "std": norm_t.std.float().numpy().tolist(),
                    "inplace": False,
                }
            )

            if self.image_resize_strategy == "resize-naive":
                self.tvf_resize_params[idx]["size"] = (resize_t.size, resize_t.size)
            elif self.image_resize_strategy == "letterbox":
                self.tvf_do_letterbox = True
                self.tvf_letterbox_fill = tuple(int(x * 255) for x in self.means[idx])
            elif self.image_resize_strategy == "resize-crop":
                pass
            else:
                raise ValueError(f"Image resize strategy {self.image_resize_strategy!r} is not supported.")

        super().__init__(**kwargs)

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        if self.tvf_do_letterbox:
            if self.tvf_letterbox_fill is None:
                raise ValueError("Letterbox fill value is not initialized.")
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        images_t: list[torch.Tensor] = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            img_idx_t = TVF.to_tensor(img_idx)
            img_idx_t = TVF.normalize(img_idx_t, **self.tvf_normalize_params[idx])
            images_t.append(img_idx_t)
        return torch.vstack(images_t)

    def preprocess(
        self,
        images: Image.Image | list[Image.Image],
        return_tensors: str | TensorType | None = None,
        **_: str,
    ) -> BatchFeature:
        if not isinstance(images, list):
            images = [images]
        pixel_values = torch.stack([self.apply_transform(image.convert("RGB")) for image in images])
        return BatchFeature(data={"pixel_values": pixel_values.float().numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Image.Image | list[Image.Image], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[list[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: ImageProcessingMixin | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput],
        images: Image.Image | list[Image.Image],
        padding: bool | str | PaddingStrategy = False,
        truncation: bool | str | TruncationStrategy | None = None,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = TensorType.PYTORCH,
    ) -> BatchFeature:
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs.")
        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    def batch_decode(self, sequences, skip_special_tokens: bool = False, clean_up_tokenization_spaces=None, **kwargs):
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces=None, **kwargs):
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> list[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


@ProcessorStepRegistry.register(name="openvla_gripper_post_processor")
class OpenVLAGripperPostProcessorStep(ActionProcessorStep):
    """Invert the gripper action to match the LIBERO environment convention.

    The OpenVLA model outputs gripper in [-1, +1] where +1 = open, -1 = close.
    LIBERO uses the opposite convention: -1 = open, +1 = close.
    """

    def action(self, action: PolicyAction) -> PolicyAction:
        action = action.clone()
        action[..., -1] = -action[..., -1]
        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_openvla_pre_post_processors(
    config: OpenVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Create thin LeRobot pre/post-processing wrappers for OpenVLA.

    The LeRobot pipeline stays responsible for transition/batch conversion, device moves,
    and dataset-stat normalization. OpenVLA-specific prompt and image processing lives in
    the local `PrismaticProcessor` classes above and is consumed from the policy/modeling side.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        OpenVLAGripperPostProcessorStep(),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
