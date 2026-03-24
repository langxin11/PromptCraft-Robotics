import base64
import math

import airsim
import cv2
import numpy as np

objects_dict = {
    "turbine1": "BP_Wind_Turbines_C_1",
    "turbine2": "StaticMeshActor_2",
    "solarpanels": "StaticMeshActor_146",
    "crowd": "StaticMeshActor_6",
    "car": "StaticMeshActor_10",
    "tower1": "SM_Electric_trellis_179",
    "tower2": "SM_Electric_trellis_7",
    "tower3": "SM_Electric_trellis_8",
}


class AirSimWrapper:
    """AirSim 多旋翼控制与视觉采集封装。"""

    def __init__(self):
        """初始化 AirSim 客户端并接管无人机控制权。"""
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def takeoff(self):
        """控制无人机起飞。"""
        self.client.takeoffAsync().join()

    def land(self):
        """控制无人机降落。"""
        self.client.landAsync().join()

    def get_drone_position(self):
        """获取无人机当前位置。

        Returns:
            list[float]: 按 [x, y, z] 返回当前位置。
        """
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def fly_to(self, point):
        """飞行到目标坐标。

        Args:
            point (list[float] | tuple[float, float, float]): 目标坐标 [x, y, z]。
                当 z 为正时会自动转换为 AirSim 的 NED 坐标系。
        """
        if point[2] > 0:
            self.client.moveToPositionAsync(point[0], point[1], -point[2], 5).join()
        else:
            self.client.moveToPositionAsync(point[0], point[1], point[2], 5).join()

    def fly_path(self, points):
        """按路径点序列飞行。

        Args:
            points (list[list[float] | tuple[float, float, float]]): 路径点集合。
                每个点格式为 [x, y, z]。
        """
        airsim_points = []
        for point in points:
            if point[2] > 0:
                airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
            else:
                airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()

    def set_yaw(self, yaw):
        """设置无人机航向角。

        Args:
            yaw (float): 航向角（度）。
        """
        self.client.rotateToYawAsync(yaw, 5).join()

    def get_yaw(self):
        """获取无人机当前航向角。

        Returns:
            float: 航向角（弧度）。
        """
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return yaw

    def get_position(self, object_name):
        """获取场景对象位置。

        Args:
            object_name (str): 预定义对象名称，需存在于 objects_dict 中。

        Returns:
            list[float]: 按 [x, y, z] 返回对象位置。
        """
        query_string = objects_dict[object_name] + ".*"
        object_names_ue = []
        while len(object_names_ue) == 0:
            object_names_ue = self.client.simListSceneObjects(query_string)
        pose = self.client.simGetObjectPose(object_names_ue[0])
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def get_scene_image_base64(self, camera_name="0", image_type=airsim.ImageType.Scene, jpeg_quality=100):
        """抓取相机画面并编码为 JPEG Base64。

        Args:
            camera_name (str): 相机名称，默认使用 "0"。
            image_type (airsim.ImageType): 图像类型，默认 Scene。
            jpeg_quality (int): JPEG 质量，范围通常为 0-100。

        Returns:
            str: JPEG 图片的 Base64 文本。

        Raises:
            RuntimeError: 当图像采集或编码失败时抛出。
        """
        response = self.client.simGetImages([
            airsim.ImageRequest(camera_name, image_type, False, False)
        ])[0]

        if response.width == 0 or response.height == 0 or len(response.image_data_uint8) == 0:
            raise RuntimeError("Failed to capture image from AirSim camera.")

        img_1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        # AirSim uncompressed Scene image is already suitable for OpenCV encode path.
        img_bgr = img_1d.reshape(response.height, response.width, 3)

        encoded, buffer = cv2.imencode(
            ".jpg",
            img_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not encoded:
            raise RuntimeError("Failed to encode image as JPEG.")

        return base64.b64encode(buffer.tobytes()).decode("ascii")
