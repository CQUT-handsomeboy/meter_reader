from abc import ABC, abstractclassmethod

import numpy as np
import cv2


class BaseReader(ABC):
    def __init__(
        self,
        dial_detection_weight_path: str,
        tilt_correction_weight_path: str,
        pointer_detect_weigt_path: str,
        result_conversion_parameters: dict,
        meta_tri_matrix: np.array,
        meta_pointer_vector: np.array,
    ) -> None:
        """基础仪表读数器，实现基本读表逻辑

        Args:
            dial_detection_weight_path (str):  表盘检测权重文件路径
            tilt_correction_weight_path (str): 倾斜矫正权重文件路径
            pointer_detect_weigt_path (str): 指针检测权重文件路径
            result_conversion_parameters (dict): 结果转换参数
            meta_tri_matrix (np.array): 元三点矩阵
            meta_pointer_vector (np.array): 元指针向量
        """
        self.dial_detection_weight_path = dial_detection_weight_path
        self.tilt_correction_weight_path = tilt_correction_weight_path
        self.pointer_detect_weigt_path = pointer_detect_weigt_path
        self.meta_tri_matrix = meta_tri_matrix
        self.meta_pointer_vector = meta_pointer_vector

        self.mini_number = result_conversion_parameters["mini_number"]  # 最小读数
        self.max_number = result_conversion_parameters["max_number"]  # 最大读数
        self.angle_mileage = result_conversion_parameters["angle_mileage"]  # 角度里程
        self.unit = result_conversion_parameters["unit"]  # 单位

        self._load_weights()

    @abstractclassmethod
    def _load_weights(self):
        """加载模型权重文件"""
        ...

    def read(self, image: np.array) -> tuple[float, str]:
        """仪表读数

        Args:
            image (np.array): 仪表图像

        Returns:
            tuple[float,str]: 二维元组，第一个元素是读数，第二个元素是单位
        """
        x1, y1, x2, y2 = self._detect_dial(image)
        dial = image[y1:y2, x1:x2]
        tri_matrix = self._tilt_correct(dial)
        standard_dial = BaseReader.transform_image(
            dial, self.meta_tri_matrix, tri_matrix
        )
        pointer_vector = self._detect_pointer(standard_dial)
        angle = BaseReader.calculate_abs_angle(self.meta_pointer_vector, pointer_vector)
        result = self._convert_result(angle)
        return result, self.unit

    @abstractclassmethod
    def _detect_dial(self, image: np.array) -> np.array:
        """检测表盘

        Args:
            image (np.array): 仪表图像

        Returns:
            np.array: 返回xyxy格式的ROI坐标
        """
        ...

    @abstractclassmethod
    def _tilt_correct(self, dial: np.array) -> np.array:
        """倾斜矫正

        Args:
            dial (np.array): 表盘图像

        Returns:
            np.array: 三点矩阵
        """
        ...

    @abstractclassmethod
    def _detect_pointer(self, standard_dial: np.array) -> np.array:
        """表针检测

        Args:
            standard_dial (np.array): 标准表盘图像

        Returns:
            np.array: 表针向量
        """
        ...

    @staticmethod
    def calculate_abs_angle(
        meta_pointer_vector: np.array, pointer_vector: np.array
    ) -> float:
        """计算指针偏转角度(顺时针)

        Args:
            meta_pointer_vector (np.array) : 元指针向量
            pointer_vector (np.array): 指针向量

        Returns:
            float: 指针偏转角度
        """
        dot_product = np.dot(meta_pointer_vector, pointer_vector)
        norm_v1 = np.linalg.norm(meta_pointer_vector)
        norm_v2 = np.linalg.norm(pointer_vector)
        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = -1 if cos_angle < -1 else (1 if cos_angle > 1 else cos_angle)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg

    @staticmethod
    def transform_image(
        image: np.array,
        meta_tri_matrix: np.array,
        tri_matrix: np.array,
    ) -> np.array:
        """仿射变换

        Args:
            image (np.array): 仪表图片
            meta_tri_matrix (np.array): 元三点矩阵
            tri_matrix (np.array): 三点矩阵

        Returns:
            np.array: 经过仿射变换后的图像
        """
        assert tri_matrix.shape == meta_tri_matrix.shape == (3, 2), "三点矩阵或元三点矩阵尺寸错误"

        meta_tri_matrix = meta_tri_matrix.astype(np.float32)
        tri_matrix = tri_matrix.astype(np.float32)
        image = image.astype(np.float32)

        trans = cv2.getAffineTransform(tri_matrix, meta_tri_matrix)

        image = cv2.warpAffine(image, trans, image.shape)
        image = image.astype(np.uint8)

        return image

    def _convert_result(self, angle: float) -> float:
        """转换读数结果

        Args:
            angle (float): 角度

        Returns:
            float: 读数结果
        """
        result = (
            self.max_number - self.mini_number
        ) / self.angle_mileage * angle + self.mini_number
        return result
