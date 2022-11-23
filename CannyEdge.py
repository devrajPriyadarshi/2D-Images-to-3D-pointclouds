import os
import random
import numpy as np
import cv2
from matplotlib import pyplot as plt


def CannyEdgeDetection(img):
    edges = cv2.Canny(img, 50, 100)
    return edges

if __name__ == "__main__":
    
    data_folder = "./data/renders"

    sub_folder = data_folder + "/" + str(random.choice( os.listdir( data_folder)))
    sub_sub_folder = sub_folder + "/" + str(random.choice( os.listdir( sub_folder)))
    img_file = sub_sub_folder + "/render_"+ str(random.randint(0,4)) + ".png"

    img = cv2.imread(img_file)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    edges = CannyEdgeDetection(img)
    cv2.imshow("img", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()