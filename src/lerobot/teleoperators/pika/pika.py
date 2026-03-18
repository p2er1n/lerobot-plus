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

import logging
import math
import threading
from typing import Any

import numpy as np

from lerobot.types import RobotAction

from ..teleoperator import Teleoperator
from .config_pika import PikaTeleopConfig

logger = logging.getLogger(__name__)

# Joint name mapping: ROS joint index -> lerobot joint name
# Position array: [joint1, joint2, joint3, joint4, joint5, joint6, gripper]
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift", 
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",  # Index 5 in position array, but we treat it separately
]

# Index of gripper in the position array (last element)
GRIPPER_INDEX = 6


class Pika(Teleoperator):
    """
    Pika teleoperator that receives joint states from ROS.
    
    Subscribes to sensor_msgs/JointState messages on a ROS topic and converts
    them to lerobot actions. The position array should contain:
    - positions[0:6]: joint angles in radians
    - positions[6]: gripper opening distance in meters
    """

    config_class = PikaTeleopConfig
    name = "pika"

    def __init__(self, config: PikaTeleopConfig):
        super().__init__(config)
        self.config = config
        
        # Thread-safe storage for latest joint states
        self._lock = threading.Lock()
        self._latest_positions: np.ndarray | None = None
        self._connected = False
        
        # ROS subscriber (initialized in connect)
        self._subscriber = None
        self._ros_node = None

    @property
    def action_features(self) -> dict[str, type]:
        """Return the structure of actions this teleoperator produces."""
        # Match the format expected by SO-100/101 and similar robots
        features = {}
        for joint in JOINT_NAMES:
            features[f"{joint}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Pika does not support feedback."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        """Pika does not require calibration."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Connect to ROS and subscribe to joint states topic."""
        try:
            import rospy
            from sensor_msgs.msg import JointState
        except ImportError as e:
            raise ImportError(
                "ROS packages not found. Please install rospy and sensor_msgs: "
                "sudo apt install ros-noetic-rospy ros-noetic-sensor-msgs"
            ) from e

        # Initialize ROS node (anonymous to allow multiple instances)
        if not rospy.core.is_initialized():
            rospy.init_node('pika_teleop', anonymous=True)
        
        # Subscribe to joint states
        self._subscriber = rospy.Subscriber(
            self.config.ros_topic,
            JointState,
            self._joint_state_callback,
            queue_size=1
        )
        
        self._connected = True
        logger.info(f"Pika teleop connected, listening on {self.config.ros_topic}")

    def _joint_state_callback(self, msg):
        """ROS callback: receive and store joint states."""
        if len(msg.position) < 7:
            logger.warning(
                f"Received JointState with {len(msg.position)} positions, expected at least 7"
            )
            return
        
        with self._lock:
            self._latest_positions = np.array(msg.position[:7])

    def calibrate(self) -> None:
        """Pika does not require calibration."""
        logger.info("Pika teleop does not require calibration")

    def configure(self) -> None:
        """No additional configuration needed."""
        pass

    def get_action(self) -> RobotAction:
        """
        Get the current action from the Pika device.
        
        Converts ROS joint states to lerobot action format:
        - Joint angles: radians -> degrees (if use_degrees=True)
        - Gripper: meters -> normalized 0-100 scale
        """
        with self._lock:
            if self._latest_positions is None:
                # No data received yet, return zeros
                logger.warning("No joint states received yet, returning zero action")
                positions = np.zeros(7)
            else:
                positions = self._latest_positions.copy()
        
        action = {}
        
        # Process 6 joint angles
        for i, joint_name in enumerate(JOINT_NAMES[:-1]):  # Exclude gripper
            angle_rad = positions[i]
            
            if self.config.use_degrees:
                # Convert radians to degrees
                angle_deg = math.degrees(angle_rad)
                action[f"{joint_name}.pos"] = float(angle_deg)
            else:
                action[f"{joint_name}.pos"] = float(angle_rad)
        
        # Process gripper (index 6)
        # Note: JOINT_NAMES[5] is "gripper", but gripper is at index 6 in positions
        gripper_m = positions[GRIPPER_INDEX]
        
        # Normalize gripper: 0 (fully closed) to gripper_scale (fully open)
        # gripper_m is in meters, scale to 0-gripper_scale
        gripper_normalized = (gripper_m / self.config.gripper_max_m) * self.config.gripper_scale
        gripper_normalized = np.clip(gripper_normalized, 0, self.config.gripper_scale)
        
        action["gripper.pos"] = float(gripper_normalized)
        
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Pika does not support feedback."""
        pass

    def disconnect(self) -> None:
        """Disconnect from ROS."""
        if self._subscriber is not None:
            self._subscriber.unregister()
            self._subscriber = None
        
        self._connected = False
        logger.info("Pika teleop disconnected")
