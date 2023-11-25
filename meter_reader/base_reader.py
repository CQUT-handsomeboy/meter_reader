from abc import ABCMeta, abstractclassmethod
import numpy as np


class BaseReader(meta=ABCMeta):
    def __init__(
        self,
        dial_detection_weight_path: str,
        tilt_correction_weight_path: str,
        angle_recognition_weight_path: str,
        result_conversion_parameters: dict,
        meta_tri_matrix: np.array,
        meta_pointer_vector: np.array,
    ) -> None:
        """基础仪表读数器，实现基本读表逻辑

        Args:
            dial_detection_weight_path (str):  表盘检测权重文件路径
            tilt_correction_weight_path (str): 倾斜矫正权重文件路径
            angle_recognition_weight_path (str): 角度识别权重文件路径
            result_conversion_parameters (dict): 结果转换参数
            meta_tri_matrix (np.array): 元三点矩阵
            meta_pointer_vector (np.array): 元指针向量
        """
        self.dial_detection_weight_path = dial_detection_weight_path
        self.tilt_correction_weight_path = tilt_correction_weight_path
        self.angle_recognition_weight_path = angle_recognition_weight_path
        self.result_conversion_parameters = result_conversion_parameters
        self.meta_tri_matrix = meta_tri_matrix
        self.meta_pointer_vector = meta_pointer_vector