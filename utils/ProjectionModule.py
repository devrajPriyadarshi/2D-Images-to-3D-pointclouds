import os
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt

import scipy.io
import plotly.graph_objects as go
import numpy as np
import math

def CannyEdgeDetection(img):
    edges = cv2.Canny(img, 50, 100)
    return edges

def quaternion_to_euler(x, y, z, w):
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  X = math.degrees(math.atan2(t0, t1))

  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  Y = math.degrees(math.asin(t2))

  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  Z = math.degrees(math.atan2(t3, t4))

  return X, Y, Z

def gaussianBlock(sigma, k = 5):
    arr = np.zeros((k,k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            arr[i][j] = math.exp(-math.sqrt((i-k//2)**2 + (j-k//2)**2)/(2*sigma))
    return arr


def eul2rot(theta) :
    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])
    return R

def DifferentialProjectionModule(s,ss, writeOutput = False):
    
    data_folder = "./data/renders"

    sub_folder = s
    sub_sub_folder = ss
    img_file = data_folder + "/" + sub_folder + "/" + sub_sub_folder + "/render_"+ str(random.randint(0,4)) + ".png"

    data_folder = "./data/gt/downsampled"
    mat_file = data_folder + "/" + sub_folder + "/" + sub_sub_folder + ".mat"
    img = cv2.imread(img_file)
    edges = CannyEdgeDetection(img)
    if not writeOutput:
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow("img", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite("./testOutputs/"+ss+"_image.jpg", img)
        cv2.imwrite("./testOutputs/"+ss+"_edges.jpg", edges)

    mat = scipy.io.loadmat(mat_file)
    p = mat['points']
    x, y, z = [ x[0] for x in p], [ x[1] for x in p], [ x[2] for x in p]
    if not writeOutput:
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
        fig.show()

    fx = fy = 120
    u0 = v0 = 32
    # R = np.array([  [ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]])
    R = eul2rot([0,0,0])
    K = np.array([  [ fx, 0, u0], [ 0, fy, v0], [ 0, 0, 1]])
    T = np.array([  [ 0], [0 ], [ 2.5]])
    ext = np.array(np.bmat([R, T]))
    # print(R)
    # print(T)
    # print(K)
    # print(ext)

    new_p = []
    for x in p:
        x = np.array( [ [x[0]], [x[1]], [x[2]], [1] ])
        # print(x)
        y = K@(ext@x)
        y = np.array([ y[0], y[1], y[2] ])
        new_p.append([y[0][0]/y[2][0], y[1][0]/y[2][0], 1])

    xmean = ymean = n = 0
    for p in new_p:
        xmean += p[0]
        ymean += p[1]
        n+=1
    
    xmean /= n
    ymean /= n

    print(xmean)
    print(ymean)

    x, y, z = [ x[0] for x in new_p], [ x[1] for x in new_p], [ x[2] for x in new_p]
    if not writeOutput:
        fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
        fig.show()

    proj_arr = np.zeros((64,64),dtype=np.float32)
    k = 5
    g = gaussianBlock(sigma=0.2, k=10)
    proj_W = proj_H = 64

    for p in new_p:
        x = round(p[0])
        y = round(p[1])

        if (x>=0 and x<= proj_W) and (y>=0 and y<= proj_H):
            arr_ = np.zeros((64,64),dtype=np.float32)
            for i in range(0,k):
                for j in range(0,k):
                    if (x-(i+k//2)>=0 and x-(i+k//2)<= proj_W) and (y-(j+k//2)>=0 and y-(j+k//2)<= proj_H):
                        arr_[x-(i+k//2)][y-(j+k//2)] = g[i][j]
            proj_arr = proj_arr + arr_

    # if np.amax(proj_arr) > 255:
        # proj_arr_ = np.round(255*proj_arr/np.amax(proj_arr))
    # else:
        # proj_arr_ = np.round(proj_arr)
    proj_arr_ = np.round(255*proj_arr/np.amax(proj_arr))
    proj_img = proj_arr_.astype(np.uint8)

    # cv2.imshow("img", proj_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if writeOutput:
        cv2.imwrite("./testOutputs/"+ss+"_proj.jpg", proj_img)
    else:
        cv2.imshow("img", proj_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    s = ["02958343","03001627","02958343", "02958343", "02691156"]
    ss = ["19d35f3e0c0b402125ddb89f02fe6cc0", "583deb9e5cf939a9daeb838d0771f3b5", "a74c1aa71334c6af20363e2561dd589a", "8fadf13734ff86b5f9e6f9c7735c6b41", "ed4aaf81dc577bedac4f72bf08dc79a6" ]

    for i in range(len(s)):
        DifferentialProjectionModule(s[i], ss[i], writeOutput = True)


    # DifferentialProjectionModule(s[1], ss[1], writeOutput=False)