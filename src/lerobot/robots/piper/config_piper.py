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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    """Configuration for Piper robot arm."""

    can_interface: str = "can0"
    bitrate: int = 1_000_000

    # Piper SDK returns 6 joints; keep order stable
    joint_names: list[str] = field(default_factory=lambda: [f"joint_{i+1}" for i in range(6)])

    # Optional sign flips applied symmetrically to obs/actions (length must match joints)
    joint_signs: list[int] = field(default_factory=lambda: [-1, 1, 1, -1, 1, -1])

    # Allow teleop joints (e.g., SO101) to reference Piper joints directly by name
    joint_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "shoulder_pan": "joint_1",
            "shoulder_lift": "joint_2",
            "elbow_flex": "joint_3",
            "wrist_flex": "joint_5",
            "wrist_roll": "joint_6",
        }
    )

    # Expose gripper as "gripper.pos" in mm if True
    include_gripper: bool = False

    # Optional cameras; leave empty when not used
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # When False, expose normalized [-100,100] joint percents; when True, degrees/mm
    use_degrees: bool = True

    # Timeout in seconds to wait for SDK EnablePiper during connect
    enable_timeout: float = 5.0
