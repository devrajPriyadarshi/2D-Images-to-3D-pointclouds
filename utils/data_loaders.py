import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import torch.utils.data
from shapenet_taxonomy import shapenet_id_to_category, shapenet_category_to_id
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from random import sample

from CannyEdge import CannyEdgeDetection

pc_dir = "../data/shapenet/ShapeNet_pointclouds/"
img_dir = "../data/shapenet/ShapeNetRendering/"

def parseTrainData():
    images = []
    models = []
    angles = []

    f = open('../data/shapenet/splits/train_models.json')
    data = json.load(f)

    labels = data.keys()
    labels = list(labels)

    labels2 = ['03001627', '04256520', '04379243'] # chair, sofa, table
    models_ = 2000
    render_ = 12
    for x in labels2:
        assert x in labels, "Error, class not found"
            

    for class_ in labels2:
        sub_folders = data[class_]
        sub_folders_ = sample(sub_folders, models_)
        for variations_ in tqdm(sub_folders_):
            pc_file = pc_dir + variations_ + "/pointcloud_1024.npy"
            # pc = np.load(pc_file)

            meta_loc = img_dir + variations_ + "/rendering/rendering_metadata.txt"
            # meta_file = open(meta_loc).readlines()
            # for ang in sample(list(range(24)), render_):
            for ang in range(10):
                if ang < 10:
                    img_file = img_dir + variations_ + "/rendering/0"+str(ang)+".png"
                else:
                    img_file = img_dir + variations_ + "/rendering/"+str(ang)+".png"
                # img = cv2.imread(img_file, cv2.COLOR_BGR2RGB)

                models.append(pc_file)
                images.append(img_file)
                # angles.append(meta_file[ang])

    return images, models, angles

def parseValData():
    images = []
    models = []
    angles = []

    f = open('../data/shapenet/splits/val_models.json')
    data = json.load(f)

    labels = data.keys()
    labels = list(labels)

    labels2 = ['03001627', '04256520', '04379243'] # chair, sofa, table
    models_ = 100
    render_ = 12
    for x in labels2:
        assert x in labels, "Error, class not found"
            

    for class_ in labels2:
        sub_folders = data[class_]
        sub_folders_ = sample(sub_folders, models_)
        for variations_ in tqdm(sub_folders_):
            pc_file = pc_dir + variations_ + "/pointcloud_1024.npy"
            # pc = np.load(pc_file)

            meta_loc = img_dir + variations_ + "/rendering/rendering_metadata.txt"
            # meta_file = open(meta_loc).readlines()
            for ang in range(24):
                if ang < 10:
                    img_file = img_dir + variations_ + "/rendering/0"+str(ang)+".png"
                else:
                    img_file = img_dir + variations_ + "/rendering/"+str(ang)+".png"
                # img = cv2.imread(img_file, cv2.COLOR_BGR2RGB)

                models.append(pc_file)
                images.append(img_file)
                # angles.append(meta_file[ang])

    return images, models, angles

class DatasetLoader(Dataset):
    def __init__(self, model_paths, image_paths, angel_paths, sourceTransform):
        self.model_path = model_paths
        self.image_path = image_paths
        self.angel_path = angel_paths
        self.sourceTransform = sourceTransform
        return

    def __len__(self):
        assert len(self.image_path) == len(self.model_path)
        return len(self.image_path)

    def __getitem__(self, idx):
               
        
        image = cv2.imread(self.image_path[idx], -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        edge = CannyEdgeDetection(image)
        gt_pc = np.load(self.model_path[idx])
        gt_pc = torch.from_numpy(gt_pc)

        image = Image.fromarray(image)
        edge = Image.fromarray(edge)

        if self.sourceTransform:
            image = self.sourceTransform(image)
            edge = self.sourceTransform(edge)

        return image, edge, gt_pc