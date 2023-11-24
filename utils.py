import cv2
import numpy as np

from sklearn.cluster import KMeans
from ultralytics import YOLO

yolo_thermometer_detect = None
yolo_thermometer_correct = None
yolo_thermometer_read = None


def load_pt(mode: str, pt_path: str) -> None:
    """根据模式加载模型文件

    Args:
        mode (str): 模式
        pt_path (str): 模型文件的位置

    Raises:
        ModeError: 模式参数错误
    """
    global yolo_thermometer_detect, yolo_thermometer_correct, yolo_thermometer_read
    match mode:
        case "detect":
            yolo_thermometer_detect = YOLO(pt_path)
        case "correct":
            yolo_thermometer_correct = YOLO(pt_path)
        case "read":
            yolo_thermometer_read = YOLO(pt_path)
        case _:
            raise type("ModeError", (Exception,), {})("模式参数错误")


# 使用Kmeans聚类获取掩膜的中心坐标
get_mask_center = lambda mask: np.reshape(
    KMeans(n_clusters=1, n_init=10)
    .fit(np.array(np.where(mask != 0)).transpose()[:, ::-1])
    .cluster_centers_,
    -1,
)


def gt_abs_angle(v1: np.array, v2: np.array) -> float:
    """获取两个向量的夹角(度)

    Args:
        v1 (np.array): 向量1的坐标
        v2 (np.array): 向量2的坐标

    Returns:
        float: 两向量夹角(度)
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = -1 if cos_angle < -1 else (1 if cos_angle > 1 else cos_angle)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def run_yolo(mode: str, image: np.array):
    """根据不同模式执行不同操作

    Args:
        mode (str): 模式 包括检测仪表模式、检测特征点模式和检测表针模式
        image (np.array): numpy数组形式的图片文件

    Returns:
        np.array:(positions) xyxy形式的ROI区域坐标
        np.array:(pts_image) 图像特征点矩阵
        float:(angle) 指针偏转角度
    """
    match mode:
        case "detect":  # 检测仪表,返回xyxy坐标
            data = yolo_thermometer_detect(image)[0].boxes.data
            if len(data) == 0:
                return
            positions = data[0, :4].reshape(-1, 2).cpu().numpy().astype(np.int16)
            return positions
        case "correct":  # 检测特征点,返回图像三点矩阵
            data = yolo_thermometer_correct(image)[0].boxes.data
            if len(data) == 0 or len(data) != 3:
                return
            data = data.cpu().numpy()
            data = np.concatenate(
                [
                    data[data[:, -1] == 0],
                    data[data[:, -1] == 1],
                    data[data[:, -1] == 2],
                ]
            )[:, :4]
            pts_image = ((data[:, :2] + data[:, 2:]) / 2).astype(np.int16)
            return pts_image
        case "read":  # 检测表针,返回角度
            data = yolo_thermometer_read(image)[0].masks.data.cpu().numpy()
            if len(data) == 2:
                temperature_pointer_mask, humidity_pointer_mask = data
                if np.count_nonzero(temperature_pointer_mask) < np.count_nonzero(
                    humidity_pointer_mask
                ):
                    temperature_pointer_mask, humidity_pointer_mask = (
                        humidity_pointer_mask,
                        temperature_pointer_mask,
                    )
            else:
                temperature_pointer_mask = data[0]

            image_shape = np.array(image.shape[:2])
            mask_shape = np.array(temperature_pointer_mask.shape)
            template_shape = np.array([500, 500])

            image_mask_ratio = image_shape / mask_shape
            image_template_ratio = image_shape / template_shape

            temperature_pointer_mask_center = get_mask_center(temperature_pointer_mask)
            temperature_pointer_center = np.array([250, 250])
            temperature_pointer_origin_center = np.array([142, 361])

            temperature_pointer_mask_center = (
                temperature_pointer_mask_center * image_mask_ratio
            )
            temperature_pointer_center = (
                temperature_pointer_center * image_template_ratio
            )
            temperature_pointer_origin_center = (
                temperature_pointer_origin_center * image_template_ratio
            )

            cv2.circle(
                image,
                temperature_pointer_mask_center.astype(np.int16),
                10,
                (0, 255, 0),
                -1,
            )
            cv2.circle(
                image, temperature_pointer_center.astype(np.int16), 10, (0, 0, 255), -1
            )

            cv2.line(
                image,
                temperature_pointer_center.astype(np.int16),
                temperature_pointer_mask_center.astype(np.int16),
                (0, 255, 0),
                10,
            )
            cv2.line(
                image,
                temperature_pointer_center.astype(np.int16),
                temperature_pointer_origin_center.astype(np.int16),
                (0, 255, 0),
                10,
            )

            pointer_vector = (
                temperature_pointer_mask_center - temperature_pointer_center
            )
            pointer_origin_vector = (
                temperature_pointer_origin_center - temperature_pointer_center
            )

            # pointer_vertical_vector 为标准的指针掩膜中心原始坐标
            pointer_vertical_vector = np.array([-666, -648])

            a = gt_abs_angle(pointer_origin_vector, pointer_vector)
            b = gt_abs_angle(pointer_vertical_vector, pointer_vector)

            if a >= 90 and b >= 90:
                x = 360 - a
            elif (a >= 90 and b <= 90) or (a <= 90 and b <= 90):
                x = a

            angle = x / 180 * 55 + (-25)

            return angle


def _transform_image(
    template_shape_2d: tuple,
    image: np.array,
    pts_template,
    pts_image,
) -> np.array:
    """_summary_

    Args:
        template_shape_2d (tuple): 模版2d尺寸
        image (np.array): 图像
        pts_template: 模版三点矩阵
        pts_image: 图像三点矩阵

    Returns:
        np.array: 仿射变换后的图像
    """
    pts_image = np.float32(pts_image)
    pts_template = np.float32(pts_template)

    assert pts_image.shape == (3, 2) and pts_template.shape == (3, 2), "未识别到完整特征点"
    assert len(template_shape_2d) == 2

    trans = cv2.getAffineTransform(pts_image, pts_template)

    image = image.astype(np.float32)

    image = cv2.warpAffine(image, trans, template_shape_2d)
    image = image.astype(np.uint8)

    return image


def main(image:np.array):
    # detect
    positions = run_yolo("detect", image)
    assert positions is not None, "未检测到仪表盘"
    pt1, pt2 = positions
    image = image[pt1[1] : pt2[1], pt1[0] : pt2[0]]
    # correct
    pts_image = run_yolo("correct", image)

    # 模版三点矩阵 标准图像上的原始坐标
    pts_template = ((877, 2377), (1498, 1080), (2145, 1892))

    image = _transform_image(
        (3000, 3000), image, pts_template, pts_image
    )
    # read
    res = run_yolo("read", image)

    return res


if __name__ == "__main__":
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        res = main(frame)
        res = cv2.resize(res, (500, 500))
        cv2.imwrite("res.png", res)
        cv2.imshow("win", res)
        cv2.waitKey(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()
