import os
import cv2
import csv
import numpy as np

dir1 = ""

k = 0
for i in range(10):

    dir2 = dir1+str(i)+"/"
    f_ = os.listdir(dir2)
    for j in range(len(f_)):
        # print(j, dir2, f_[j])
        f = dir2+f_[j]

        img = cv2.imread(f, 0)
        img_res = cv2.resize(img, (28,28))

        cv2.imwrite(dir1+"res/"+str(i)+"_"+str(j)+".png", img_res)


