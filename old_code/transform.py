import cv2
import numpy as np
import sys

pts1 = np.float32([(278,230),(210,199),(137,156)])  
pts2 = np.float32([(125,207),(169,151),(107,104)])

# 得到仿射变换矩阵
trans = cv2.getAffineTransform(pts2,pts1)

img2 = cv2.imread(sys.argv[1])
img2 = cv2.resize(img2,(0,0),fx=0.125,fy=0.125,interpolation=cv2.INTER_NEAREST)
img1 = cv2.warpAffine(img2,trans,img2.shape[:2])

cv2.imshow("img2",img2)
cv2.imshow("img1",img1)

cv2.waitKey(0)