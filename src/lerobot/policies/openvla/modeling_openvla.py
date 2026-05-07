#!/usr/bin/env python

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np
import torch
import torch.nn as nn
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING

from ..pretrained import ActionSelectKwargs, PreTrainedPolicy
from .configuration_openvla import (
    LLM_BACKBONE_TO_HF_PATH,
    TIMM_OVERRIDE_ACT_LAYER,
    VALID_LLM_BACKBONES,
    VALID_VISION_BACKBONES,
    VISION_BACKBONE_TO_RESOLUTION,
    VISION_BACKBONE_TO_TIMM_ID,
    OpenVLAConfig,
)
from .processor_openvla import PrismaticImageProcessor, PrismaticProcessor


logger = logging.getLogger(__name__)
IGNORE_INDEX = -100
_EMPTY_ACTION_PROMPT_TOKEN_ID = 29871
LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama",
    "llama2-13b-pure": "llama",
    "llama2-7b-chat": "llama",
    "llama2-13b-chat": "llama",
    "vicuna-v15-7b": "llama",
    "vicuna-v15-13b": "llama",
    "mistral-v0.1-7b-pure": "mistral",
    "mistral-v0.1-7b-instruct": "mistral",
    "phi-2-3b": "phi",
}


def unpack_tuple(fn: Callable[[Any], tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: nn.Module):
    if not hasattr(ls_module, "gamma"):
        return
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, type(ls_module))
    del ls_module.gamma


class PrismaticHFConfig(PretrainedConfig):
    model_type: str = "prismatic"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "vicuna-v15-7b",
        arch_specifier: str = "no-align+gelu-mlp",
        use_fused_vision_backbone: bool | None = None,
        image_resize_strategy: str = "letterbox",
        text_config: dict[str, Any] | None = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: Any,
    ) -> None:
        if vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(f"Vision backbone {vision_backbone_id!r} not in {sorted(VALID_VISION_BACKBONES)}")
        if llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(f"LLM backbone {llm_backbone_id!r} not in {sorted(VALID_LLM_BACKBONES)}")

        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(vision_backbone_id.startswith(v) for v in ["dinoclip", "dinosiglip"])
        )
        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[vision_backbone_id]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[vision_backbone_id]
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTION[vision_backbone_id]
        self.image_resize_strategy = image_resize_strategy
        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[llm_backbone_id]
        self.llm_max_length = llm_max_length
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

        metaclass = LLM_BACKBONE_TO_HF_METACLASS[llm_backbone_id]
        self.text_config = CONFIG_MAPPING[metaclass](**text_config) if text_config is not None else CONFIG_MAPPING[metaclass]()
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class OpenVLAHFConfig(PrismaticHFConfig):
    model_type: str = "openvla"

    def __init__(
        self,
        norm_stats: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None,
        n_action_bins: int = 256,
        **kwargs: Any,
    ) -> None:
        self.norm_stats = norm_stats
        self.n_action_bins = n_action_bins
        super().__init__(**kwargs)

    @classmethod
    def from_lerobot_config(cls, cfg: OpenVLAConfig) -> "OpenVLAHFConfig":
        return cls(
            vision_backbone_id=cfg.vision_backbone_id,
            llm_backbone_id=cfg.llm_backbone_id,
            use_fused_vision_backbone=cfg.use_fused_vision_backbone,
            image_resize_strategy=cfg.image_resize_strategy,
            llm_max_length=cfg.llm_max_length,
            pad_token_id=cfg.pad_token_id,
            pad_to_multiple_of=cfg.pad_to_multiple_of,
            n_action_bins=cfg.n_action_bins,
            norm_stats=cfg.norm_stats,
        )


class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: list[int],
        timm_model_ids: list[str],
        timm_override_act_layers: list[str | None],
    ) -> None:
        super().__init__()
        try:
            import timm
            from timm.models.vision_transformer import LayerScale
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PrismaticVisionBackbone requires timm. Install timm in the lerobot-plus environment to use the local OpenVLA model implementation."
            ) from exc

        self.use_fused_vision_backbone = use_fused_vision_backbone
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 fused vision backbones."
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        for module in self.featurizer.modules():
            if module.__class__.__name__ == "LayerScale" or isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if module.__class__.__name__ == "LayerScale" or isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
        return torch.cat([patches, patches_fused], dim=2)


class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)
        return projected_features


@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    projector_features: torch.FloatTensor | None = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticHFConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True
    _no_split_modules: ClassVar[list[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True
    _supports_sdpa: bool = True
    _supports_cache_class: bool = True
    _supports_static_cache: bool = True
    _supports_quantized_cache: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel, GenerationMixin):
    config_class: PretrainedConfig = PrismaticHFConfig

    def __init__(self, config: PrismaticHFConfig) -> None:
        super().__init__(config)

        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone,
            config.image_sizes,
            config.timm_model_ids,
            config.timm_override_act_layers,
        )
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        attn_implementation = getattr(config, "_attn_implementation", None)
        lm_kwargs = {"attn_implementation": attn_implementation} if attn_implementation is not None else {}
        self.language_model = AutoModelForCausalLM.from_config(config.text_config, **lm_kwargs)
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.post_init()
        self.generation_config = getattr(self.language_model, "generation_config", None)
        if self.generation_config is None:
            self.generation_config = GenerationConfig.from_model_config(self.language_model.config)

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        try:
            self.language_model.tie_weights(*args, **kwargs)
        except TypeError:
            self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: int | None = None, pad_to_multiple_of: int | None = None):
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings
        return updated_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_projector_features: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> tuple | PrismaticCausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache and not self.training
        projected_patch_embeddings = None

        if input_ids is not None and input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is currently supported only for batch size 1."
            assert past_key_values is not None, "past_key_values are required during cached generation."
            assert labels is None, "labels are not expected during cached generation."
            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing input_ids in language-only forward."
            assert past_key_values is None, "past_key_values are not expected in language-only forward."
            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif (input_ids is not None and input_ids.shape[0] == pixel_values.shape[0]) or (
            inputs_embeds is not None and inputs_embeds.shape[0] == pixel_values.shape[0]
        ):
            assert past_key_values is None, "past_key_values are not expected during multimodal forward."
            patch_features = self.vision_backbone(pixel_values)
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            input_embeddings = self.get_input_embeddings()(input_ids)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            raise ValueError("Invalid PrismaticForConditionalGeneration forward() call for the given inputs.")

        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings
            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | list[torch.FloatTensor] | None]:
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported.")
        if past_key_values is not None and input_ids is not None:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )
        # Pass pixel_values only on the first generation step (prefill).
        # In subsequent decoding steps, image features are already cached in past_key_values.
        # Note: in transformers>=5.x, generate() initializes an empty DynamicCache before the first
        # forward call, so we must check is_initialized rather than relying on past_key_values is None.
        is_first = past_key_values is None
        if hasattr(past_key_values, "is_initialized"):
            is_first = is_first or not past_key_values.is_initialized
        if is_first and pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs

    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAHFConfig

    def __init__(self, config: OpenVLAHFConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    @staticmethod
    def _check_unnorm_key(norm_stats: dict[str, dict[str, Any]], unnorm_key: str | None) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                "This OpenVLA model was trained on more than one dataset; pass an explicit unnorm_key. "
                f"Available keys: {sorted(norm_stats.keys())}"
            )
            unnorm_key = next(iter(norm_stats.keys()))
        assert unnorm_key in norm_stats, f"unnorm_key must be one of {sorted(norm_stats.keys())}"
        return unnorm_key

    def get_action_dim(self, unnorm_key: str | None = None) -> int:
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: str | None = None) -> dict[str, Any]:
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def predict_action(
        self,
        input_ids: torch.LongTensor | None = None,
        unnorm_key: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Predict a single action from tokenized input.

        Uses manual autoregressive decoding instead of ``self.generate()``.
        HF transformers >=5.x changes the prefill mechanism (initializes an
        empty cache before the first forward call, and validates kwargs via
        ``_validate_model_kwargs``), making it unreliable for custom vision-
        language models unless fully registered with the transformers library.
        """
        if input_ids is None:
            raise ValueError("predict_action requires input_ids.")

        # Append the special empty-action prompt token if not already present.
        if not torch.all(input_ids[:, -1] == _EMPTY_ACTION_PROMPT_TOKEN_ID):
            input_ids = torch.cat(
                (
                    input_ids,
                    torch.full(
                        (input_ids.shape[0], 1),
                        _EMPTY_ACTION_PROMPT_TOKEN_ID,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    ),
                ),
                dim=1,
            )
            if "attention_mask" in kwargs and isinstance(kwargs["attention_mask"], torch.Tensor):
                kwargs["attention_mask"] = torch.cat(
                    [
                        kwargs["attention_mask"],
                        torch.ones(
                            (kwargs["attention_mask"].shape[0], 1),
                            dtype=kwargs["attention_mask"].dtype,
                            device=kwargs["attention_mask"].device,
                        ),
                    ],
                    dim=1,
                )

        action_dim = self.get_action_dim(unnorm_key)

        # Extract multimodal inputs from kwargs.
        pixel_values = kwargs.pop("pixel_values", None)
        attention_mask = kwargs.pop("attention_mask", None)

        # ---- Step 1: multimodal forward (image + text) ----
        outputs = self.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        generated_tokens: list[torch.Tensor] = [next_token]

        # ---- Steps 2..N: autoregressive decoding with KV cache ----
        past_key_values = outputs.past_key_values
        for _ in range(action_dim - 1):
            outputs = self.forward(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

        predicted_action_token_ids = torch.cat(generated_tokens, dim=-1)[0, :action_dim].detach().cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )


def _resolve_torch_dtype(dtype_name: str, device_type: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    dtype = dtype_map[dtype_name]
    if device_type == "cpu" and dtype is not torch.float32:
        return torch.float32
    return dtype


def _to_batched_action_tensor(actions: Sequence[np.ndarray | Tensor | Sequence[float]], device: torch.device) -> Tensor:
    tensors: list[Tensor] = []
    for action in actions:
        tensor = torch.as_tensor(action, dtype=torch.float32, device=device)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        tensors.append(tensor.reshape(-1))
    return torch.stack(tensors, dim=0)


def _resolve_auto_model_class():
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    except ImportError:
        pass
    try:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except ImportError as exc:
        raise ImportError(
            "OpenVLA requires a multimodal auto model class from transformers. Neither AutoModelForImageTextToText nor AutoModelForVision2Seq is available."
        ) from exc


class OpenVLAPolicy(PreTrainedPolicy):
    config_class = OpenVLAConfig
    name = "openvla"

    def __init__(self, config: OpenVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.processor = None
        self.model = None
        self._backend_loaded = False
        self._input_dtype = torch.float32
        self._bin_centers = self._compute_bin_centers(config.n_action_bins)
        self.reset()
        # Load the backend model eagerly so that PEFT adapter loading (via PeftModel.from_pretrained)
        # can find target modules for LoRA injection.
        self._ensure_backend_loaded()

    @staticmethod
    def _compute_bin_centers(n_action_bins: int) -> np.ndarray:
        bins = np.linspace(-1, 1, n_action_bins)
        return (bins[:-1] + bins[1:]) / 2.0

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _validate_peft_config(self, peft_config) -> None:
        """Allow PEFT even when pretrained_path is None, since OpenVLA loads
        base weights from model_id rather than a lerobot checkpoint path."""
        if not self.config.pretrained_path and not self.config.model_id:
            raise ValueError(
                "PEFT requires either pretrained_path or model_id to be set "
                "so that base model weights are available for fine-tuning."
            )

    def wrap_with_peft(self, peft_config=None, peft_cli_overrides=None):
        """Override to ensure the model is loaded before PEFT wraps it."""
        return super().wrap_with_peft(peft_config=peft_config, peft_cli_overrides=peft_cli_overrides)

    def reset(self):
        self._last_instruction: str | list[str] | None = None
        self._last_action: Tensor | None = None

    def _target_device(self) -> torch.device:
        return torch.device(self.config.device or "cpu")

    def _load_local_processor(self) -> PrismaticProcessor:
        import json

        preprocessor_path = Path(self.config.model_id) / "preprocessor_config.json"
        with open(preprocessor_path) as f:
            pre_cfg = json.load(f)

        input_sizes = [tuple(size) for size in pre_cfg["input_sizes"]]
        means = [tuple(mean) for mean in pre_cfg["means"]]
        stds = [tuple(std) for std in pre_cfg["stds"]]
        interpolations = pre_cfg.get("interpolations")

        image_processor = PrismaticImageProcessor(
            use_fused_vision_backbone=pre_cfg["use_fused_vision_backbone"],
            image_resize_strategy=pre_cfg["image_resize_strategy"],
            input_sizes=input_sizes,
            interpolations=interpolations,
            means=means,
            stds=stds,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        return PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

    def _load_local_openvla_backend(self, model_dtype: torch.dtype, attn_implementation: str | None):
        import json
        from safetensors.torch import load_file

        base_path = Path(self.config.model_id)
        hf_config = OpenVLAHFConfig.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            local_files_only=self.config.local_files_only,
        )
        if hf_config.norm_stats is None and self.config.norm_stats is not None:
            hf_config.norm_stats = self.config.norm_stats
        hf_config._attn_implementation = attn_implementation

        model = OpenVLAForActionPrediction(hf_config)
        if model_dtype != torch.float32:
            model = model.to(dtype=model_dtype)

        index_path = base_path / "model.safetensors.index.json"
        if index_path.is_file():
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            checkpoint_keys = set(index["weight_map"].keys())
            unexpected_keys = set()
            for shard_name in shard_files:
                shard_state = load_file(str(base_path / shard_name), device="cpu")
                if model_dtype != torch.float32:
                    shard_state = {
                        k: v.to(dtype=model_dtype) if torch.is_floating_point(v) else v
                        for k, v in shard_state.items()
                    }
                incompatible = model.load_state_dict(shard_state, strict=False, assign=True)
                unexpected_keys.update(incompatible.unexpected_keys)
                del shard_state
            model_keys = set(model.state_dict().keys())
            missing_keys = sorted(model_keys - checkpoint_keys)
            unexpected_keys = sorted(unexpected_keys | (checkpoint_keys - model_keys))

        else:
            state_dict = load_file(str(base_path / "model.safetensors"), device="cpu")
            if model_dtype != torch.float32:
                state_dict = {
                    k: v.to(dtype=model_dtype) if torch.is_floating_point(v) else v
                    for k, v in state_dict.items()
                }
            incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
            missing_keys = incompatible.missing_keys
            unexpected_keys = incompatible.unexpected_keys
            del state_dict

        if missing_keys:
            logger.warning("Missing keys when loading local OpenVLA backend: %s", missing_keys[:20])
        if unexpected_keys:
            logger.warning("Unexpected keys when loading local OpenVLA backend: %s", unexpected_keys[:20])
        return model

    def _load_openvla_backend(self) -> None:
        # Resolve relative model_id against the config file directory.
        model_id = Path(self.config.model_id)
        if not model_id.is_absolute():
            # self.config.pretrained_path points to the lerobot config directory (or checkpoint).
            cfg_dir = (
                Path(self.config.pretrained_path).resolve()
                if self.config.pretrained_path
                else Path.cwd()
            )
            self.config.model_id = str((cfg_dir / model_id).resolve())
            logger.info("Resolved relative model_id=%s to %s", model_id, self.config.model_id)

        device = self._target_device()
        device_type = device.type
        model_dtype_name = self.config.torch_dtype if device_type == "cpu" else self.config.gpu_torch_dtype
        attn_implementation = (
            self.config.attn_implementation if device_type == "cpu" else self.config.gpu_attn_implementation
        )
        model_dtype = _resolve_torch_dtype(model_dtype_name, device_type)

        processor = self._load_local_processor()

        model = None
        local_error = None
        try:
            model = self._load_local_openvla_backend(
                model_dtype=model_dtype,
                attn_implementation=attn_implementation,
            )
            logger.info("Loaded OpenVLA backend from local Prismatic port.")
        except Exception as exc:
            local_error = exc
            logger.warning("Falling back to HF auto model path for OpenVLA backend: %s", exc)

        if model is None:
            auto_model_class = _resolve_auto_model_class()
            model_kwargs = {
                "revision": self.config.revision,
                "trust_remote_code": self.config.trust_remote_code,
                "local_files_only": self.config.local_files_only,
                "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
                "torch_dtype": model_dtype,
            }
            if attn_implementation:
                model_kwargs["attn_implementation"] = attn_implementation
            model = auto_model_class.from_pretrained(self.config.model_id, **model_kwargs)
            if local_error is not None:
                logger.info("OpenVLA auto-model fallback succeeded after local port failed: %s", local_error)

        model = model.to(device)
        model.eval()
        self.processor = processor
        self.model = model
        self._input_dtype = model_dtype
        self._backend_loaded = True

    def _ensure_backend_loaded(self) -> None:
        if not self._backend_loaded:
            self._load_openvla_backend()

    def _get_norm_stats(self) -> dict[str, Any] | None:
        model_config = getattr(getattr(self, "model", None), "config", None)
        if model_config is not None and getattr(model_config, "norm_stats", None) is not None:
            return model_config.norm_stats
        return self.config.norm_stats

    def _check_unnorm_key(self, unnorm_key: str | None) -> str:
        norm_stats = self._get_norm_stats()
        if not norm_stats:
            raise ValueError("OpenVLA action unnormalization requires norm_stats, but none are available.")
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    "This OpenVLA model has multiple normalization-stat entries; pass policy.unnorm_key explicitly. "
                    f"Available keys: {sorted(norm_stats.keys())}"
                )
            unnorm_key = next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(f"unnorm_key={unnorm_key!r} is not available. Valid keys: {sorted(norm_stats.keys())}")
        return unnorm_key

    def _get_action_stats(self, unnorm_key: str | None) -> dict[str, Any]:
        norm_stats = self._get_norm_stats()
        if not norm_stats:
            raise ValueError("OpenVLA action statistics are unavailable because norm_stats is missing.")
        resolved_key = self._check_unnorm_key(unnorm_key)
        return norm_stats[resolved_key]["action"]

    def _get_action_dim(self, unnorm_key: str | None) -> int:
        return len(self._get_action_stats(unnorm_key)["q01"])

    def _get_action_vocab_size(self) -> int:
        assert self.model is not None
        model_config = getattr(self.model, "config", None)
        if model_config is None:
            raise ValueError("Loaded OpenVLA backend does not expose a config object.")
        text_config = getattr(model_config, "text_config", None)
        if text_config is None or not hasattr(text_config, "vocab_size"):
            raise ValueError("Loaded OpenVLA backend config is missing text_config.vocab_size.")
        pad_multiple = getattr(model_config, "pad_to_multiple_of", self.config.pad_to_multiple_of)
        return text_config.vocab_size - pad_multiple

    def _move_processor_inputs(self, inputs: Any) -> Any:
        device = self._target_device()
        if hasattr(inputs, "to") and not isinstance(inputs, Mapping):
            try:
                return inputs.to(device)
            except TypeError:
                pass
        if isinstance(inputs, Mapping):
            moved = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    moved[key] = value.to(device=device, dtype=self._input_dtype) if value.is_floating_point() else value.to(device=device)
                else:
                    moved[key] = value
            return moved
        return inputs

    def _extract_instruction(self, batch: dict[str, Tensor | Any]) -> list[str]:
        if self.config.task_feature_key not in batch:
            raise KeyError(f"Expected task feature {self.config.task_feature_key} in batch. Available keys: {sorted(batch.keys())}")
        value = batch[self.config.task_feature_key]
        if isinstance(value, str):
            return [value]
        if isinstance(value, bytes):
            return [value.decode()]
        if isinstance(value, Sequence) and not isinstance(value, (torch.Tensor, np.ndarray, str, bytes)):
            return [item.decode() if isinstance(item, bytes) else str(item) for item in value]
        raise TypeError(f"Unsupported instruction container type for key {self.config.task_feature_key}: {type(value)}")

    def _tensor_to_pil(self, image: Tensor) -> Image.Image:
        image = image.detach().cpu()
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor with 3 dims [C,H,W], got shape={tuple(image.shape)}")
        if image.shape[0] in {1, 3}:
            image = image.permute(1, 2, 0)
        elif image.shape[-1] not in {1, 3}:
            raise ValueError(f"Unsupported image tensor shape for OpenVLA adapter: {tuple(image.shape)}")
        image = image.clamp(0, 1).mul(255).round().to(torch.uint8) if image.is_floating_point() else image.to(torch.uint8)
        array = image.numpy()
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return Image.fromarray(array).convert("RGB")

    def _extract_images(self, batch: dict[str, Tensor | Any]) -> list[Image.Image]:
        image_key = self.config.image_feature_key
        if image_key is None:
            raise ValueError("image_feature_key should be resolved during config.validate_features().")
        if image_key not in batch:
            raise KeyError(f"Expected image feature {image_key} in batch. Available keys: {sorted(batch.keys())}")
        value = batch[image_key]
        if isinstance(value, Image.Image):
            return [value.convert("RGB")]
        if isinstance(value, np.ndarray):
            if value.ndim == 3:
                return [Image.fromarray(value.astype(np.uint8)).convert("RGB")]
            if value.ndim == 4:
                return [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in value]
            raise ValueError(f"Unsupported numpy image batch shape: {value.shape}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unsupported image container type for key {image_key}: {type(value)}")
        if value.ndim == 3:
            return [self._tensor_to_pil(value)]
        if value.ndim == 4:
            return [self._tensor_to_pil(frame) for frame in value]
        if value.ndim == 5:
            return [self._tensor_to_pil(frame[-1]) for frame in value]
        raise ValueError(f"Unsupported image tensor batch shape for OpenVLA adapter: {tuple(value.shape)}")

    def _format_prompt(self, instruction: str) -> str:
        return self.config.prompt_template.format(instruction=instruction)

    def _prepare_openvla_inputs(self, instruction: str, image: Image.Image) -> Any:
        assert self.processor is not None
        return self._move_processor_inputs(self.processor(self._format_prompt(instruction), image, return_tensors="pt"))

    def _predict_action_from_generate(self, inputs: Mapping[str, Any], unnorm_key: str | None) -> np.ndarray:
        assert self.model is not None
        if "input_ids" not in inputs:
            raise ValueError("OpenVLA generate fallback requires input_ids in processor outputs.")
        input_ids = inputs["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"Expected input_ids to be a tensor, got {type(input_ids)}")
        if not torch.all(input_ids[:, -1] == _EMPTY_ACTION_PROMPT_TOKEN_ID):
            appended = torch.full((input_ids.shape[0], 1), _EMPTY_ACTION_PROMPT_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat((input_ids, appended), dim=1)
            inputs = dict(inputs)
            inputs["input_ids"] = input_ids
            if "attention_mask" in inputs and isinstance(inputs["attention_mask"], torch.Tensor):
                attention_mask = inputs["attention_mask"]
                one = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                inputs["attention_mask"] = torch.cat((attention_mask, one), dim=1)
        action_dim = self._get_action_dim(unnorm_key)
        generated_ids = self.model.generate(**inputs, max_new_tokens=action_dim)
        predicted_action_token_ids = generated_ids[0, -action_dim:].detach().cpu().numpy()
        discretized_actions = self._get_action_vocab_size() - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self._bin_centers.shape[0] - 1)
        normalized_actions = self._bin_centers[discretized_actions]
        action_stats = self._get_action_stats(unnorm_key)
        mask = action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool))
        action_high = np.array(action_stats["q99"])
        action_low = np.array(action_stats["q01"])
        return np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)

    def _run_openvla_inference(self, inputs: Any, unnorm_key: str | None) -> np.ndarray | Tensor | Sequence[float]:
        assert self.model is not None
        # self.model is OpenVLAForActionPrediction (PEFT wraps the outer OpenVLAPolicy, not this inner model).
        # Its predict_action() now delegates to self.generate() which correctly handles
        # multimodal prefill + cached decoding via prepare_inputs_for_generation().
        if hasattr(self.model, "predict_action"):
            return self.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        if not isinstance(inputs, Mapping):
            raise TypeError("OpenVLA generate fallback expects processor outputs to be a mapping when the backend lacks predict_action().")
        return self._predict_action_from_generate(inputs, unnorm_key=unnorm_key)

    def _predict_openvla_action(self, batch: dict[str, Tensor | Any]) -> Tensor:
        assert self.model is not None
        instructions = self._extract_instruction(batch)
        images = self._extract_images(batch)
        if len(instructions) == 1 and len(images) > 1:
            instructions = instructions * len(images)
        if len(images) != len(instructions):
            raise ValueError(f"Mismatched OpenVLA batch inputs: {len(images)} images vs {len(instructions)} instructions.")
        actions = []
        for instruction, image in zip(instructions, images, strict=True):
            inputs = self._prepare_openvla_inputs(instruction, image)
            actions.append(self._run_openvla_inference(inputs, self.config.unnorm_key))
        action_tensor = _to_batched_action_tensor(actions, device=self._target_device())
        expected_dim = self.config.action_feature.shape[0] if self.config.action_feature is not None else None
        if expected_dim is not None and action_tensor.shape[-1] != expected_dim:
            raise ValueError(f"OpenVLA returned action_dim={action_tensor.shape[-1]}, expected action_dim={expected_dim}.")
        self._last_instruction = instructions
        self._last_action = action_tensor
        return action_tensor

    def forward(self, batch: dict[str, Tensor], *args, **kwargs) -> tuple[Tensor, dict | None]:
        assert self.model is not None
        assert self.processor is not None

        # --- 1. Extract inputs from batch ---
        instructions = self._extract_instruction(batch)
        images = self._extract_images(batch)  # list of PIL images
        actions = batch["action"].to(device=self._target_device(), dtype=torch.float32)
        actions = actions.view(actions.shape[0], -1)  # ensure [B, action_dim]

        # Broadcast single instruction to all images
        if len(instructions) == 1 and len(images) > 1:
            instructions = instructions * len(images)

        # --- 2. Process images through PrismaticImageProcessor ---
        pixel_values = self.processor.image_processor(images, return_tensors="pt")["pixel_values"]
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(device=self._target_device(), dtype=model_dtype)

        # --- 3. Build prompts and tokenize ---
        prompts = [self._format_prompt(inst) for inst in instructions]
        tokenizer = self.processor.tokenizer
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        prompt_ids = prompt_inputs.input_ids.to(self._target_device())

        # Append the special empty-action prompt token (29871)
        separator = torch.full(
            (prompt_ids.shape[0], 1), _EMPTY_ACTION_PROMPT_TOKEN_ID,
            dtype=prompt_ids.dtype, device=self._target_device(),
        )
        prompt_ids = torch.cat([prompt_ids, separator], dim=1)

        # --- 4. Normalize actions to [-1, 1] using q01/q99 ---
        action_stats = self.model.get_action_stats(self.config.unnorm_key)
        q01 = torch.tensor(action_stats["q01"], device=actions.device, dtype=actions.dtype)
        q99 = torch.tensor(action_stats["q99"], device=actions.device, dtype=actions.dtype)
        mask = torch.tensor(
            action_stats.get("mask", [True] * actions.shape[-1]),
            device=actions.device, dtype=torch.bool,
        )
        denom = q99 - q01
        denom = torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)
        normalized = 2.0 * (actions - q01) / denom - 1.0
        normalized = torch.where(mask, normalized, actions)
        normalized = torch.clamp(normalized, -1.0, 1.0)

        # --- 5. Discretize actions into token IDs ---
        n_bins = self.config.n_action_bins
        vocab_size = self.model.vocab_size
        discretized = torch.round((normalized + 1.0) / 2.0 * (n_bins - 1)).long()
        discretized = torch.clamp(discretized, 0, n_bins - 2)  # n_bins-2 = 254
        action_token_ids = vocab_size - discretized - 1

        # --- 6. Build full input_ids and labels ---
        input_ids = torch.cat([prompt_ids, action_token_ids], dim=1)

        # Labels: IGNORE for prompt, action tokens for the action portion
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[:, prompt_ids.shape[1]:] = action_token_ids

        attention_mask = torch.ones_like(input_ids)

        # --- 7. Forward through model ---
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        loss = outputs.loss
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        return self.select_action(batch, **kwargs).unsqueeze(1)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: ActionSelectKwargs) -> Tensor:
        self.eval()
        return self._predict_openvla_action(batch)
