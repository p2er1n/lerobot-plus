#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import OBS_STATE

VISION_BACKBONE_TO_RESOLUTION: dict[str, list[int]] = {
    "clip-vit-l": [224],
    "siglip-vit-so400m": [224],
    "dinov2-vit-l": [224],
    "in1k-vit-l": [224],
    "clip-vit-l-336px": [336],
    "siglip-vit-so400m-384px": [384],
    "dinoclip-vit-l-336px": [336, 336],
    "dinosiglip-vit-so-224px": [224, 224],
    "dinosiglip-vit-so-384px": [384, 384],
}
VISION_BACKBONE_TO_TIMM_ID: dict[str, list[str]] = {
    "clip-vit-l": ["vit_large_patch14_clip_224.openai"],
    "clip-vit-l-336px": ["vit_large_patch14_clip_336.openai"],
    "dinov2-vit-l": ["vit_large_patch14_reg4_dinov2.lvd142m"],
    "in1k-vit-l": ["vit_large_patch16_224.augreg_in21k_ft_in1k"],
    "siglip-vit-so400m": ["vit_so400m_patch14_siglip_224"],
    "siglip-vit-so400m-384px": ["vit_so400m_patch14_siglip_384"],
    "dinoclip-vit-l-336px": [
        "vit_large_patch14_reg4_dinov2.lvd142m",
        "vit_large_patch14_clip_336.openai",
    ],
    "dinosiglip-vit-so-224px": [
        "vit_large_patch14_reg4_dinov2.lvd142m",
        "vit_so400m_patch14_siglip_224",
    ],
    "dinosiglip-vit-so-384px": [
        "vit_large_patch14_reg4_dinov2.lvd142m",
        "vit_so400m_patch14_siglip_384",
    ],
}
TIMM_OVERRIDE_ACT_LAYER: dict[str, list[str | None]] = {
    "clip-vit-l": ["quick_gelu"],
    "clip-vit-l-336px": ["quick_gelu"],
    "dinov2-vit-l": [None],
    "in1k-vit-l": [None],
    "siglip-vit-so400m": [None],
    "siglip-vit-so400m-384px": [None],
    "dinoclip-vit-l-336px": [None, "quick_gelu"],
    "dinosiglip-vit-so-224px": [None, None],
    "dinosiglip-vit-so-384px": [None, None],
}
LLM_BACKBONE_TO_HF_PATH: dict[str, str] = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf",
    "llama2-13b-pure": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "vicuna-v15-7b": "lmsys/vicuna-7b-v1.5",
    "vicuna-v15-13b": "lmsys/vicuna-13b-v1.5",
    "mistral-v0.1-7b-pure": "mistralai/Mistral-7B-v0.1",
    "mistral-v0.1-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
    "phi-2-3b": "microsoft/phi-2",
}
VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION)
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH)


@PreTrainedConfig.register_subclass("openvla")
@dataclass
class OpenVLAConfig(PreTrainedConfig):
    """OpenVLA adapter configuration for LeRobot.

    This keeps the LeRobot-facing policy contract while importing the high-level
    HF/Prismatic semantics that matter for OpenVLA:
    - vision backbone identity and resize strategy,
    - text backbone identity and max sequence length,
    - dataset normalization statistics metadata,
    - LoRA and runtime knobs copied from upstream finetune/inference flows.
    """

    n_obs_steps: int = 1
    chunk_size: int = 1
    n_action_steps: int = 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    model_id: str = "openvla/openvla-7b"
    revision: str | None = None
    trust_remote_code: bool = True
    local_files_only: bool = False

    image_feature_key: str | None = None
    task_feature_key: str = "task"
    use_processor_prompt: bool = True
    prompt_template: str = "In: What action should the robot take to {instruction}?\nOut:"
    unnorm_key: str | None = None

    vision_backbone_id: str = "siglip-vit-so400m"
    llm_backbone_id: str = "vicuna-v15-7b"
    use_fused_vision_backbone: bool | None = None
    image_resize_strategy: str = "letterbox"
    llm_max_length: int = 2048
    pad_token_id: int = 32000
    pad_to_multiple_of: int = 64
    n_action_bins: int = 256
    norm_stats: dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None = None

    torch_dtype: str = "float32"
    attn_implementation: str = "eager"
    low_cpu_mem_usage: bool = True
    compile_model: bool = False
    gpu_torch_dtype: str = "bfloat16"
    gpu_attn_implementation: str = "flash_attention_2"

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    save_steps: int = 5_000
    shuffle_buffer_size: int = 100_000

    max_state_dim: int = 32
    max_action_dim: int = 32

    optimizer_lr: float = 5e-4
    optimizer_weight_decay: float = 0.0

    timm_model_ids: list[str] = field(init=False, repr=False)
    timm_override_act_layers: list[str | None] = field(init=False, repr=False)
    image_sizes: list[int] = field(init=False, repr=False)
    hf_llm_id: str = field(init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

        if self.n_obs_steps != 1:
            raise ValueError("OpenVLA adapter currently only supports n_obs_steps=1.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot be greater than chunk_size.")
        if self.chunk_size != 1:
            raise ValueError("OpenVLA adapter scaffold currently assumes chunk_size=1.")
        if self.vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(
                f"vision_backbone_id={self.vision_backbone_id!r} is not in {sorted(VALID_VISION_BACKBONES)}"
            )
        if self.llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(f"llm_backbone_id={self.llm_backbone_id!r} is not in {sorted(VALID_LLM_BACKBONES)}")
        if self.image_resize_strategy not in {"resize-naive", "resize-crop", "letterbox"}:
            raise ValueError("image_resize_strategy must be one of: resize-naive, resize-crop, letterbox.")
        if self.torch_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("torch_dtype must be one of: float32, float16, bfloat16.")
        if self.gpu_torch_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("gpu_torch_dtype must be one of: float32, float16, bfloat16.")
        if self.llm_max_length <= 0:
            raise ValueError("llm_max_length must be positive.")
        if self.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive.")
        if self.n_action_bins <= 0:
            raise ValueError("n_action_bins must be positive.")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive.")
        if self.lora_dropout < 0:
            raise ValueError("lora_dropout cannot be negative.")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("grad_accumulation_steps must be positive.")
        if self.save_steps <= 0:
            raise ValueError("save_steps must be positive.")
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive.")
        if not self.task_feature_key:
            raise ValueError("task_feature_key must be a non-empty feature name.")
        if "{instruction}" not in self.prompt_template:
            raise ValueError("prompt_template must contain the placeholder {instruction}.")

        self.use_fused_vision_backbone = (
            self.use_fused_vision_backbone
            if self.use_fused_vision_backbone is not None
            else any(self.vision_backbone_id.startswith(prefix) for prefix in ["dinoclip", "dinosiglip"])
        )
        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[self.vision_backbone_id]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTION[self.vision_backbone_id]
        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[self.llm_backbone_id]

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("OpenVLA adapter expects at least one visual input feature.")
        if self.action_feature is None:
            raise ValueError("OpenVLA adapter expects an action output feature.")

        if self.image_feature_key is None:
            image_keys = sorted(self.image_features.keys())
            self.image_feature_key = image_keys[0]
        elif self.image_feature_key not in self.image_features:
            raise ValueError(
                f"image_feature_key={self.image_feature_key!r} is not among available visual features: "
                f"{sorted(self.image_features.keys())}"
            )

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]

    @property
    def reward_delta_indices(self) -> None:
        return None
