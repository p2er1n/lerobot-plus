#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@dataclass
class PikaConfig:
    """Base configuration class for Pika teleoperator."""

    # ROS topic to subscribe for joint states
    ros_topic: str = "/joint_states_gripper"

    # Whether to convert radians to degrees for joint positions
    # Set to True if target robot uses degrees (like SO-100/101)
    use_degrees: bool = True

    # Gripper scale factor: convert meters to robot gripper units
    # For SO-100/101, gripper uses 0-100 range
    gripper_scale: float = 100.0

    # Gripper max opening in meters (used for normalization)
    gripper_max_m: float = 0.08  # 8cm max opening


@TeleoperatorConfig.register_subclass("pika")
@dataclass
class PikaTeleopConfig(TeleoperatorConfig, PikaConfig):
    pass
