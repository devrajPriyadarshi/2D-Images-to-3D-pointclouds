import torch
import torch.nn as nn

import numpy as np
import math
from math import pi
from random import uniform

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

def normalizePC(pc_batch):

    pc_norm = []
    for pc in pc_batch:
        pc_np = pc.to("cpu").numpy()
        centroid = np.mean(pc_np, axis=0)
        print(centroid.shape)
        print(centroid)
        pc_np -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(pc_np)**2,axis=-1)))
        print(furthest_distance)
        pc_np /= 2*furthest_distance
        pc_norm.append(pc_np)

    return torch.tensor(pc_norm)

def rotatePC(pc, eul = [0,0,0]):
    R = eul2rot([eul[0],eul[1],eul[2]])
    new_p = []
    for x_ in pc:
        x = x_.T
        y = R@x
        y_ = y.T
        new_p.append(y_)
    return np.array(new_p)

def projectImg(pc1, eul = [0,0,0]):

    fx = fy = 120
    u0 = v0 = 36
    R = eul2rot([eul[0],eul[1],eul[2]])
    K = np.array([  [ fx, 0, u0], [ 0, fy, v0], [ 0, 0, 1]])
    T = np.array([  [ 0], [0 ], [ 2.5]])
    ext = np.array(np.bmat([R, T]))

    new_p = []
    for x_ in pc1:
        x2 = np.array( [ [x_[0]], [x_[1]], [x_[2]], [1] ])
        y2 = K@(ext@x2)
        y2 = np.array([ y2[0], y2[1], y2[2] ])
        new_p.append([y2[0][0]/y2[2][0], y2[1][0]/y2[2][0], 1])

    proj_arr = np.zeros((64,64),dtype=np.float32)
    k = 5
    g = gaussianBlock(sigma=0.2, k=10)
    proj_W = proj_H = 64

    for p in new_p:
        x4 = round(p[0])
        y4 = round(p[1])

        if (x4>=0 and x4<= proj_W) and (y4>=0 and y4<= proj_H):
            arr_ = np.zeros((64,64),dtype=np.float32)
            for i in range(0,k):
                for j in range(0,k):
                    if (x4-(i+k//2)>=0 and x4-(i+k//2)<= proj_W) and (y4-(j+k//2)>=0 and y4-(j+k//2)<= proj_H):
                        arr_[x4-(i+k//2)][y4-(j+k//2)] = g[i][j]
            proj_arr = proj_arr + arr_

    proj_arr = proj_arr/np.amax(proj_arr)
    # proj_img = proj_arr.astype(np.uint8)

    return proj_arr


class projectionLoss(nn.Module):

    def __init__(self):
        super(projectionLoss, self).__init__()

        self.bce = torch.nn.BCELoss(reduction="sum")

    def forward(self, pc1_batch, pc2_batch):
        loss = 0
        for i in range(pc1_batch.shape[0]):
            pc1 = pc1_batch[i].detach().numpy()
            pc2 = pc2_batch[i].detach().numpy()
            Rotations = [
                # [uniform(0, pi), uniform(0, pi), uniform(0, pi)],
                # [uniform(0, -pi), uniform(0, -pi), uniform(0, -pi)],
                [uniform(0, 2*pi), uniform(0, 2*pi), uniform(0, 2*pi)],
                [uniform(-pi, pi), uniform(-pi, pi), uniform(-pi, pi)]
            ]
            for rot in Rotations:
                edge1 = projectImg(pc1, rot)
                edge2 = projectImg(pc2, rot)

                edge1 = torch.tensor(edge1.flatten(), requires_grad=True)
                edge2 = torch.tensor(edge2.flatten(), requires_grad=True)

                loss += self.bce(edge1, edge2)

        return loss