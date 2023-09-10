import cv2
import numpy as np

from ultralytics import YOLO

yolo_thermometer_detect = YOLO("thermometer_detect.pt")
yolo_thermometer_correct = YOLO("thermometer_correct.pt")
yolo_thermometer_read = YOLO("thermometer_read.pt")

def run_yolo(mode:str,image:np.array):
    match mode:
        case "detect": # 检测出仪表,返回切片
            data = yolo_thermometer_detect(image)[0].boxes.data
            if len(data) == 0:
                return
            return data[0,:4].reshape(-1,2).transpose(0,1).cpu().numpy().astype(np.int16)
        case "correct": # 检测特征点,返回图像三点矩阵
            data = yolo_thermometer_correct(image)[0]\
                .boxes.data
            if len(data) == 0 or len(data) != 3:
                return
            data = data.cpu().numpy()
            data = np.concatenate([
                data[ data[:,-1] == 0 ],
                data[ data[:,-1] == 1 ],
                data[ data[:,-1] == 2 ],
            ])[:,:4]
            pts_image = ((data[:,:2] + data[:,2:]) / 2).astype(np.int16)
            return pts_image 
        case "read":
            data = yolo_thermometer_read(image)[0].boxes.data
            

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

    image = cv2.warpAffine(image,trans,template_shape_2d)
    image = image.astype(np.uint8)

    return image

def main(image):
    # detect
    ROI_slices = run_yolo("detect", image)
    image = image[ROI_slices[0,0]:ROI_slices[0,1],
                  ROI_slices[1,0]:ROI_slices[1,1]]
    # correct
    pts_image = run_yolo("correct", image)
    image = _transform_image(
        (3000,3000),image,
        ((877, 2377), (1498, 1080), (2145, 1892)),
        pts_image
    )
    # read

    return image

if __name__ == "__main__":
    pass