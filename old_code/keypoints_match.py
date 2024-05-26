import cv2
import sys
import random
import numpy as np

# 两张图对应关键点的相对坐标差值
DELTA_X1 = 50
DELTA_Y1 = 50

# 同一张图一个点和其他点相对坐标差值
DELTA_X2 = 50
DELTA_Y2 = 50

img1 = cv2.imread(sys.argv[1])
img2 = cv2.imread(sys.argv[2])

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT_create(nfeatures=500,nOctaveLayers=3,sigma=8)

kps1, des1 = sift.detectAndCompute(gray1, None)
kps2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 用于存储左图坐标
matched_1 = []
# 用于存储右图坐标
matched_2 = []

# check两张图像的对应点的相对位置
def double_keypoints_check(x1,y1,x2,y2):
  print("double: x1 - x2 = {delta_x},y1 - y2 = {delta_y}".format(delta_x = abs(x1-x2),delta_y=abs(y1-y2) ))
  if abs(x1 - x2) <= DELTA_X1 and abs(y1 -y2) <= DELTA_Y1:
    return True
  else:
    return False

# check同一张图像一个点和另外两个点的相对位置
def triple_keypoints_check(x,y,matched):
  if len(matched) == 0:
    return True
  else:
    for match in matched:
      print("triple: x1 - x2 = {delta_x},y1 - y2 = {delta_y}".format(delta_x = abs(x-match[0]),delta_y=abs(y-match[1]) ))
      if not (abs(match[0] - x) >= DELTA_X2 and abs(match[1] - y) >= DELTA_Y2):
        return False
    return True

# 这个变量代表的是为了排除哪一个点所以要从多余这个点处开始遍历
j = 0
while True:
  for i in range(len(matches)):
    i += j
    if i > len(matches)-1:
      continue
    gray1_x = int(kps1[matches[i].queryIdx].pt[0])
    gray1_y = int(kps1[matches[i].queryIdx].pt[1])
    gray2_x = int(kps2[matches[i].trainIdx].pt[0])
    gray2_y = int(kps2[matches[i].trainIdx].pt[1])
    print("long check :",double_keypoints_check(gray1_x,gray1_y,gray2_x,gray2_y),triple_keypoints_check(gray1_x,gray2_y,matched_1),triple_keypoints_check(gray2_x,gray2_y,matched_2))
    if double_keypoints_check(gray1_x,gray1_y,gray2_x,gray2_y) and triple_keypoints_check(gray1_x,gray2_y,matched_1) and triple_keypoints_check(gray2_x,gray2_y,matched_2):
      matched_1.append((gray1_x,gray1_y))
      matched_2.append((gray2_x,gray2_y))
  if len(matched_1) >= 3:
    break
  else :
    j += 1
    matched_1,matched_2 = [],[]

for match in matched_1:
  color = random.randint(0,255),random.randint(0,255),random.randint(0,255)
  cv2.circle(img2,matched_2[matched_1.index(match)],5,color,-1)
  cv2.circle(img1,match,5,color,-1)

print(matched_1)
print(matched_2)

cv2.imshow("img1",img1)
cv2.imshow("img2",img2)

trans = cv2.getAffineTransform(np.float32(matched_2),np.float32(matched_1))
shape = img2.shape[:2]
img = cv2.warpAffine(img2,trans,shape)
cv2.imshow("img",img)


cv2.waitKey(0)