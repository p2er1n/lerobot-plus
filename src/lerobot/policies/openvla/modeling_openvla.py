#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from torch import Tensor

from ..pretrained import PreTrainedPolicy
from .configuration_openvla import OpenVLAConfig


class OpenVLAPolicy(PreTrainedPolicy):
    """Registration scaffold for an OpenVLA policy."""

    config_class = OpenVLAConfig
    name = "openvla"

    def __init__(self, config: OpenVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._is_initialized = True

    def forward(self, batch: dict[str, Tensor], *args, **kwargs) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("OpenVLA policy scaffold does not implement training yet.")

    def predict_action_chunk(self, batch: dict[str, Tensor], *args, **kwargs) -> Tensor:
        raise NotImplementedError("OpenVLA policy scaffold does not implement action chunking yet.")

    def select_action(self, batch: dict[str, Tensor], *args, **kwargs) -> Tensor:
        raise NotImplementedError("OpenVLA policy scaffold does not implement inference yet.")
