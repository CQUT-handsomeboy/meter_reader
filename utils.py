import cv2
import numpy as np

from sklearn.cluster import KMeans
from ultralytics import YOLO

yolo_thermometer_detect = YOLO("thermometer_detect.pt")
yolo_thermometer_correct = YOLO("thermometer_correct.pt")
yolo_thermometer_read = YOLO("thermometer_read.pt")

get_mask_center = lambda mask : np.reshape(KMeans(n_clusters=1,n_init=10).fit(
    np.array(np.where(mask != 0)).transpose()[:,::-1]
).cluster_centers_,-1)

# 得到两个向量0-180°的夹角
def gt_abs_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = -1 if cos_angle < -1 else (1 if cos_angle > 1 else cos_angle)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def run_yolo(mode:str,image:np.array):
    match mode:
        case "detect": # 检测仪表,返回切片
            data = yolo_thermometer_detect(image)[0].boxes.data
            if len(data) == 0:
                return
            return data[0,:4].reshape(-1,2).transpose(0,1).cpu().numpy().astype(np.int16)
        case "correct": # 检测特征点,返回图像三点矩阵
            data = yolo_thermometer_correct(image)[0].boxes.data
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
        case "read": # 检测表针,返回角度
            data = yolo_thermometer_read(image)[0].masks.data.cpu().numpy()
            if len(data) == 2:
                temperature_pointer_mask,humidity_pointer_mask = data
                if np.count_nonzero(temperature_pointer_mask) < np.count_nonzero\
                    (humidity_pointer_mask):
                    temperature_pointer_mask,humidity_pointer_mask = \
                        humidity_pointer_mask,temperature_pointer_mask
            else:
                temperature_pointer_mask = data[0]

            temperature_pointer_mask_center = get_mask_center(temperature_pointer_mask)
            ratio = np.array([3000,3000]) / np.array(temperature_pointer_mask.shape)

            temperature_pointer_origin_vector = np.array([-663,613])
            temperature_pointer_cartesian_vector = np.array([-613,-663])

            temperature_pointer_mask_center = temperature_pointer_mask_center * ratio
            temperature_pointer_center = np.array([1512,1616])
            temperature_pointer_vector = temperature_pointer_mask_center - temperature_pointer_center
            
            o_p = gt_abs_angle(temperature_pointer_origin_vector,temperature_pointer_vector)
            c_p = gt_abs_angle(temperature_pointer_cartesian_vector,temperature_pointer_vector)
            
            if o_p < 90 and c_p < 90: # 0 - 90
                print("case 1")
                return o_p
            elif o_p < 90 and c_p > 90: # 270 - 360
                print("case 2")
                return 360 - o_p
            elif o_p > 90 and c_p < 90: # 90 - 180
                print("case 3")
                return o_p
            elif o_p > 90 and c_p > 90: # 180 - 270
                print("case 4")
                return 360 - o_p

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
    angle = run_yolo("read",image)
    print(f"angle:{angle}")
    temperature = -25 + 53 / 180 * angle

    return temperature

if __name__ == "__main__":
    image = cv2.imread("./1694357518868.jpg")
    angle = main(image)
    print(f"temperature is {angle}")