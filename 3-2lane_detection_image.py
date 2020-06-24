from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import cv2
#import os
#import sys

#**********************选取关键兴趣区域，减少计算量***********************
def roi_mask(img, vertices):
  #定义mask全为黑
  mask = np.zeros_like(img)
  #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
  if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    mask_color = (255,) * channel_count
  else:
    mask_color = 255
  #将区域和图片进行填充fillPoly和叠加and
  cv2.fillPoly(mask, vertices, mask_color)
  masked_img = cv2.bitwise_and(img, mask)
  return masked_img
#**************************************************************
#**********************Hough变换检测车道线**********************
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
  lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
  line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#  draw_lines(line_img, lines)
# 划线函数，设置线型等
  draw_lanes(line_img, lines)
  return line_img

# 划线子程序
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
  for line in lines:
    for x1, y1, x2, y2 in line:
      cv2.line(img, (x1, y1), (x2, y2), color, thickness)

#**********************Hough变换后处理***********************
#********************Lane Extrapolation处理***********************
def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
  # 根据斜率正负划分某条线属于左车道还是右车道
  left_lines, right_lines = [], []
  for line in lines:
    for x1, y1, x2, y2 in line:
      k = (y2 - y1) / (x2 - x1)
      if k < 0:
        left_lines.append(line)
      else:
        right_lines.append(line)
  
  if (len(left_lines) <= 0 or len(right_lines) <= 0):
    return img

  # 移除左右偏差偏差过大的线
  clean_lines(left_lines, 0.1)
  clean_lines(right_lines, 0.1)
  left_points = [(x1, y1) for line in left_lines for x1,y1,x2,y2 in line]
  left_points = left_points + [(x2, y2) for line in left_lines for x1,y1,x2,y2 in line]
  right_points = [(x1, y1) for line in right_lines for x1,y1,x2,y2 in line]
  right_points = right_points + [(x2, y2) for line in right_lines for x1,y1,x2,y2 in line]
  
  # 分别对左右车道线的顶点集合做Linear regression
  left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
  right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])
  # 得到最终的车道线
  cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
  cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)

# 移除左右偏差偏差过大的线子程序
# 迭代计算各条直线的斜率与斜率均值的差
# 逐一移除差值过大的线
def clean_lines(lines, threshold):
  slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
  while len(lines) > 0:
    mean = np.mean(slope)
    diff = [abs(s - mean) for s in slope]
    idx = np.argmax(diff)
    if diff[idx] > threshold:
      slope.pop(idx)
      lines.pop(idx)
    else:
      break
  
# 对车道线顶点集合做Linear regression
def calc_lane_vertices(point_list, ymin, ymax):
  x = [p[0] for p in point_list]
  y = [p[1] for p in point_list]
  #做平滑拟合
  fit = np.polyfit(y, x, 1) 
  #多项式变换函数
  fit_fn = np.poly1d(fit) 
  
  xmin = int(fit_fn(ymin))
  xmax = int(fit_fn(ymax))
  
  return [(xmin, ymin), (xmax, ymax)]
#######################################################################
#                           Main                                      #
#######################################################################
#1读取图片
img = cv2.imread('.\\video_2[00_00_02][20180226-100403-1].JPG')

#****************2划定选取感兴趣的区域，减少计算量**********************
#(460, 325), (520, 325)根据相机确定的两个点
roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])

#*********************3取图，灰度化***********************************
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#*******************4高斯平滑，减少边缘干扰***************************
#gray图片输入的灰度值
#blur_ksize，高斯核大小，一般为5
#两个方向上的标准差取0
blur_ksize = 5 

blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)

#******************5利用canny算子，进行边缘检测**********************
# Canny edge detection low threshold
# Canny 边缘检测低阈值
canny_lthreshold = 50  
# Canny edge detection high threshold
# Canny 边缘检测高阈值
canny_hthreshold = 150  

edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)

#*****************6 集中到边缘检测感兴趣区域*************************
roi_edges = roi_mask(edges, roi_vtx)

#*****************7 利用Hough变换，进行直线检测**********************
# Hough变换程序
# roi_edges 边缘检测后的输入灰度图像
# rho 和 theta 对应的精度
# threshold 设定阈值，判定为直线的投票数的最小值
# min_line_len 线最短长度，小于都忽略
# max_line_gap 线最大间隔，小于认为是一条直线
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20

line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)

#****************8 将检测的直线和原图进行合成***********************
res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

cv2.namedWindow('gray')
cv2.imshow('gray', gray)
cv2.waitKey(0)

cv2.namedWindow('blur_gray')
cv2.imshow('blur_gray', blur_gray)
cv2.waitKey(0)


cv2.namedWindow('edges')
cv2.imshow('edges', edges)
cv2.waitKey(0)

cv2.namedWindow('roi_edges')
cv2.imshow('roi_edges', roi_edges)
cv2.waitKey(0)

cv2.namedWindow('res_img')
cv2.imshow('res_img', res_img)
cv2.waitKey(0)

cv2.destroyAllWindows()