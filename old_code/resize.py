'''
可以将图片的长宽都缩小到原来的1/8
'''
import cv2 as cv
import sys

filename = sys.argv[1]

img = cv.imread(filename)
  
img_test2 = cv.resize(img, (0, 0), fx=0.125, fy=0.125, interpolation=cv.INTER_AREA)

cv.imwrite(filename,img_test2)
cv.imshow('resize1', img_test2)
cv.waitKey()
cv.destroyAllWindows()