import os
import cv2
import csv
import numpy as np

dir1 = "lab/"

k = 0
for i in range(10):

    dir2 = dir1+str(i)+"/"
    f_ = os.listdir(dir2)
    for j in range(len(f_)):
        # print(j, dir2, f_[j])
        f = dir2+f_[j]

        img = cv2.imread(f, 0)
        _, img_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        img_res = cv2.resize(img_inv, (28,28))

        cv2.imwrite(dir1+"resized/"+str(i)+"_"+str(j)+".png", img_res)


