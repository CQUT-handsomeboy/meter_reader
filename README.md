# Qucik Start

```shell
git clone https://github.com/CQUT-handsomeboy/meter_reader.git
pip3 install -r requirements.txt
python3 main.py path/to/your/target/image ./LFS/thermometer_detect.pt ./LFS/tilt_correct.pt ./LFS/thermometer_read.pt
```

# 项目流程(第三版)

- 使用YOLOv8检测出仪表盘ROI。

- 对仪表盘ROI再次使用YOLOv8检测出三个特征点。

- 对检测出的三个特征点运用仿射变换进行倾斜矫正。

- 对倾斜校正后的仪表盘ROI再次使用YOLOv8检测出表针ROI。

- 取表针ROI中心，与`Standard图像`（即放射变换目标矩阵的原图像）中的指针中心作差得到表针向量。

- 将表针向量与`Standard图像`中的起始指针向量计算夹角，换算得到仪表读数。

# 项目流程(第二版)

- 使用*YOLOv8*检测出**仪表盘ROI**和**表盘种类**

- 对**仪表盘ROI**进行*特征点匹配*，使用*仿射变换*实现倾斜校正得到**倾斜校正仪表盘ROI**

- 计算得**倾斜校正仪表盘ROI中心坐标**，并对其再次使用*YOLOv8*检测出**指针ROI**和**基准点**

- 通过**倾斜校正仪表盘ROI中心坐标**和**基准点**计算出**基准直线**

- 对**指针ROI** *预处理*后使用*霍夫线检测*得到**霍夫线检测直线组**

- 将**霍夫线检测直线组**reshape后得到点坐标，使用*Kmeans聚类*聚为2类

- 将2类**Kmeans聚类坐标**分别使用*最小二乘法*拟合

- 计算两个**最小二乘法拟合直线** **相交坐标**

- 连接**相交坐标**和**倾斜校正仪表盘ROI中心坐标**，得出**指针直线**

- 联系**指针直线**和**基准直线**，求得指针偏转角度

- 根据YOLOv8识别出的**表盘种类**，得出相应读数

# 项目流程(第一版)

- 通过YOLOv3实现对表盘的分类

- 通过特征点法获得透视矩阵

- 利用透视矩阵对椭圆进行矫正，得到标准表盘

- 通过霍夫圆检测得到表盘

- 通过霍夫圆直线检测得到表针

- 计算出表针偏移方向

- 结合表盘分类实现读表

# 参考资料(第一版)

[通过透视变换进行椭圆矫正](https://blog.csdn.net/weixin_49578216/article/details/117700851?spm=1001.2014.3001.5502)

[通过特征点法来获得透视矩阵](https://blog.csdn.net/weixin_49578216/article/details/117700851?spm=1001.2014.3001.5502)

[霍夫变换直线检测](https://blog.csdn.net/leonardohaig/article/details/87907462)

[基于SURF的表针矫正](https://blog.csdn.net/weixin_49578216/article/details/117700851)

[基于Python的指针识别与表盘分析](https://blog.csdn.net/qq_44781688/article/details/118400263)

[OpenCV 视觉特征提取和匹配](https://zhuanlan.zhihu.com/p/391448297)
