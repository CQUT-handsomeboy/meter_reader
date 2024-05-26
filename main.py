from ultralytics import YOLO
from os.path import isfile

import numpy as np
import cv2
import argparse


def detect(filename: str):
    results = detect_model(filename)
    assert len(results) != 0, "未检测到仪表盘"

    for result in results:
        xyxy = tuple(int(k) for k in result.boxes.xyxy[0])
        break
    ROI = cv2.imread(filename)[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
    return ROI


def correct(roi: np.array):
    results = correct_model(roi)
    centers = {}
    assert len(results) != 0, "未检测到特征点"

    for result in results:
        xyxy = result.boxes.xyxy.numpy()
        cls = result.boxes.cls
        break
    for i, e_xyxy in enumerate(xyxy):
        center_x = int((e_xyxy[0] + e_xyxy[2]) / 2)
        center_y = int((e_xyxy[1] + e_xyxy[3]) / 2)
        cls_name = int(cls[i])  # 0 1 2
        centers[result.names[cls_name]] = (center_x, center_y)  # left up right
    assert (
        "left" in centers and "up" in centers and "right" in centers
    ), "未检测到足够的特征点"

    pts1 = np.float32([centers["right"], centers["up"], centers["left"]])
    pts2 = np.float32([(2145, 1907), (1498, 1183), (877, 2381)])

    trans = cv2.getAffineTransform(pts1, pts2)
    c_roi = cv2.warpAffine(roi, trans, (3000, 3000))

    return c_roi


def read(c_roi: np.array):
    results = read_model(c_roi)
    assert len(results) != 0, "未检测到指针"
    for result in results:
        xyxy = result.boxes.xyxy.numpy().tolist()
        break

    if len(xyxy) == 2:
        sorted(xyxy, key=lambda coord: coord[0] ** 2 + coord[1] ** 2)

    for e_xyxy in xyxy:
        center_x = int((e_xyxy[0] + e_xyxy[2]) / 2)
        center_y = int((e_xyxy[1] + e_xyxy[3]) / 2)
        break

    meta_head_pointer_center = np.array((1510, 1617))
    head_pointer_center = np.array((center_x, center_y))
    meta_head_pointer_vec = np.array([-698, 616])
    header_vec = head_pointer_center - meta_head_pointer_center

    a = calculate_directed_angle(header_vec, meta_head_pointer_vec)
    n = -25 + 80 * (a / 264.405)

    return n


def calculate_directed_angle(v1: np.array, v2: np.array):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)

    cross_product = np.cross(v1, v2)
    if cross_product > 0:
        # 如果叉积大于0，则向量2在向量1的逆时针方向
        angle_deg = 360 - angle_deg
    elif cross_product < 0:
        # 如果叉积小于0，则向量2在向量1的顺时针方向
        pass  # angle_deg已经是我们需要的角度
    else:
        # 如果叉积等于0，则两个向量共线
        # 根据具体情况来决定如何处理共线的情况
        pass

    return angle_deg


def main(filename: str):
    roi = detect(filename)
    c_roi = correct(roi)
    n = read(c_roi)
    print(f"该仪表盘读数为{n}摄氏度")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="仪表识别算法验证")
    
    parser.add_argument("file_name", type=str, help="图片文件名称")
    parser.add_argument("detect_weight_path", type=str, help="检测权重文件路径")
    parser.add_argument("correct_weight_path", type=str, help="倾斜校正权重文件路径")
    parser.add_argument("read_weight_path", type=str, help="读表权重文件路径")

    args = parser.parse_args()

    file_name = args.file_name
    detect_weight_path = args.detect_weight_path
    correct_weight_path = args.correct_weight_path
    read_weight_path = args.read_weight_path

    for path in (file_name,detect_weight_path,correct_weight_path,read_weight_path):
        if not isfile(path):
            print(f"{path}文件不存在")
            exit()

    detect_model = YOLO(detect_weight_path)
    correct_model = YOLO(correct_weight_path)
    read_model = YOLO(read_weight_path)

    main(file_name)
