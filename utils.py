import cv2
import numpy as np
import sys

from ultralytics import YOLO

def run_yolo(mode:str,image:np.array):
    if mode == "detect":
        yolo = YOLO("thermometer_detect.pt")
    elif mode == "read":
        yolo = YOLO("thermometer_read.pt")
    else:
        raise Exception("mode 参数错误")
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
    results = run_yolo("read", image)
    if len(results) != 3:
        return
    pts = [0,0,0]
    for i in range(3):
        j = str(i)
        pts[i] = ((results[j][0][0] + results[j][1][0]) // 2,
             (results[j][0][1] + results[j][1][1]) // 2)
    x = _transform_image((3000,3000),
                     image,
                     ((877, 2377), (1498, 1080), (2145, 1892)),
                     pts)
    return x

if __name__ == "__main__":
    argv = sys.argv
    assert len(argv) == 2,"请将第二个参数指定为需要进行倾斜校正的图片名称"
    filename = argv[1]
    x = cv2.imread(filename)
    x = tilt_correct(x)
    x = cv2.resize(x,(500,500))
    cv2.imshow("x",x)
    cv2.waitKey(0)