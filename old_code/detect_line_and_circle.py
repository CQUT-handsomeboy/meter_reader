import cv2
import numpy as np
import random
import sys

file_name = sys.argv[1]
src = cv2.imread(file_name)

# 使用中值滤波对图像进行预处理
srcBlur = cv2.medianBlur(src,3,0)
'''
# 替代方案 高斯滤波
srcBlur = cv2.GaussianBlur(src, (3, 3), 0)
'''
gray = cv2.cvtColor(srcBlur, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 90, 160, apertureSize=3)

my_line_args = {
  "minLineLength":0,
  "threshold":50,
  "maxLineGap":50
}

my_circle_args = {
  "param1":100,
  "param2":100, 
  "minDist":100, # 不需要动
  "minRadius":0, # 不需要动
  "maxRadius":1000, # 不需要动
  "dp":1, # 不需要动
}

# 霍夫线检测
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, **my_line_args)
# 霍夫圆检测
circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,**my_circle_args)

def get_line_args():
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, **my_line_args)
  if type(lines) != np.ndarray:
    print("None")
    return 2
  print("检测出直线条数"+str(len(lines)))
  return len(lines)

def get_circle_args():
  circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,**my_circle_args)
  if type(circles) != np.ndarray:
    print("None")
    return 2
  print("检测出圆个数"+str(len(circles)))
  return len(circles)


# 检测表针
while get_line_args() != 1:
  # 通过随机数调整参数，直到只检测出一条线
  my_line_args["threshold"] = random.randint(0,100)
  my_line_args["minLineLength"] = random.randint(0,100)
  my_line_args["maxLineGap"] = random.randint(0,10)
x1,y1,x2,y2 = cv2.HoughLinesP(edges, 1, np.pi / 180, **my_line_args)[0][0]
cv2.line(src,(x1,y1),(x2,y2),(0,255,0),3)
# 检测表盘
while get_circle_args() != 1:
  # 通过随机数调整参数，直到只检测出一个圆
  my_circle_args["param1"] = random.randint(50,150)
  my_circle_args["param2"] = random.randint(50,150)
x,y,r = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,**my_circle_args)[0][0]
x,y,r = int(x),int(y),int(r)
# 标识
cv2.circle(src,(x,y),r,(0,255,0),3)
cv2.circle(src,(x,y),5,(0,255,0),-1)

src = cv2.resize(src,(0,0),fx=0.125,fy=0.125,interpolation=cv2.INTER_NEAREST)

cv2.imshow("src",src)
cv2.waitKey(0)