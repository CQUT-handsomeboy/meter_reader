import cv2
import numpy as np

from ultralytics import YOLO

yolo_thermometer_detect = YOLO("thermometer_detect.pt")
yolo_thermometer_read = YOLO("thermometer_read.pt")

def run_yolo(mode:str,image:np.array):
    assert mode in ["detect","read"]
    yolo = yolo_thermometer_detect if mode == "detect" else yolo_thermometer_read
    results = {}
    boxes = yolo(image)[0].boxes
    for i, xywh in enumerate(boxes.xywh):
        x,y,w,h = tuple(map(lambda x:int(x),xywh))
        left_up,right_down = (x - w//2,y - w //2),(x + w//2,y + h//2)
        category_id = int(boxes.cls[i].item())
        conf = boxes.conf[i].item()
        results[str(category_id)] = \
            (left_up,right_down,conf)
    return results

def run_yolo_and_draw(mode:str,image:np.array):
    results = run_yolo(mode, image)
    for category_id,(left_up,right_down,conf) in results.items():
        cv2.rectangle(image, left_up,right_down, (0,255,0),5)
    return results

# 图像仿射变换
def _transform_image(template_shape_2d:tuple, # 模版2d尺寸
                    image:np.array, # 图像
                    pts_template, # 模版三点矩阵
                    pts_image): # 图像三点矩阵
    pts_image = np.float32(pts_image)
    pts_template = np.float32(pts_template)

    assert pts_image.shape == (3,2) and pts_template.shape == (3,2)
    assert len(template_shape_2d) == 2

    trans = cv2.getAffineTransform(pts_image,pts_template)

    image = image.astype(np.float32)

    x = cv2.warpAffine(image,trans,template_shape_2d)
    x = x.astype(np.uint8)

    return x

# 倾斜校正
def tilt_correct(image):
    detect_results = run_yolo("detect", image)
    left_up,right_down = detect_results["0"][:2]
    image = image[left_up[1]:right_down[1],left_up[0]:right_down[0]]
    read_results = run_yolo("read", image)
    if len(read_results) != 3:
        return
    pts = [0,0,0]
    for i in range(3):
        j = str(i)
        pts[i] = ((read_results[j][0][0] + read_results[j][1][0]) // 2,
             (read_results[j][0][1] + read_results[j][1][1]) // 2)
    x = _transform_image((3000,3000),
                     image,
                     ((877, 2377), (1498, 1080), (2145, 1892)),
                     pts)
    return x

if __name__ == "__main__":
    image = cv2.imread("./1.jpg")
    image = tilt_correct(image)
    image = cv2.resize(image,(500,500))
    cv2.imshow("x",image)
    cv2.waitKey(0)