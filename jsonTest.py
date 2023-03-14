import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

f = open('data/shapenet/splits/train_models.json')

data = json.load(f)

# print("\n"*5)

# print(data.keys())

x = data.keys()

x = list(x)

folder1 = x[0]

file1 = data[x[0]][0]

print(folder1)
print(file1)

dir_ = file1

pc_dir = "data/shapenet/ShapeNet_pointclouds/"
img_dir = "data/shapenet/ShapeNetRendering/"

pc_file = pc_dir + file1 + "/pointcloud_1024.npy"
pc = np.load(pc_file)

x = pc[:, 0]
y = pc[:, 1]
z = pc[:, 2]

img_file = img_dir + file1 + "/rendering/01.png"
img = cv2.imread(img_file, -1)

cv2.imshow("image", img)
cv2.waitKey(0)

fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2))])
fig.show()


print(pc.shape)
