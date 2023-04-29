import sys
sys.path.insert(0, 'RepVGG')
sys.path.insert(0, 'utils')
sys.path.insert(0, 'MitsubaRenderer')
import logging

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import plotly.graph_objects as go

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from RepVGG.repvgg import create_RepVGG_A0
from RepVGG import se_block

from auxilarynet import AuxilaryBranchCNN
from main_model import pointCloudGenerator
from pointcloudpyramid import PointCloudPyramid, Pyramid_Layer_1, Pyramid_Layer_2, Pyramid_Layer_3
# from chamferdist import ChamferDistance


from data_loaders import parseValData, DatasetLoader
from custom_losses import TotalLoss
from ProjectionLoss import projectImg, normalizePC, rotatePC

from MitsubaRenderer.render_mitsuba2_pc import main
import mitsuba as mi
from math import pi
# utils.
# utils.
# utils.
# utils.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

MainBranch = create_RepVGG_A0().to(device)
weightsA0 = torch.load("Pretrained_Networks/RepVGG-A0-train.pth")
MainBranch.load_state_dict(weightsA0)

AuxiliaryBranch = AuxilaryBranchCNN().to(device)
weightsAux = torch.load("Pretrained_Networks/Auxiliary_Network.pth")
AuxiliaryBranch.load_state_dict(weightsAux["model_state_dict"])

PCP = PointCloudPyramid(Pyramid_Layer_1(), Pyramid_Layer_2(), Pyramid_Layer_3()).to(device)

#setup logging
logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)

def testingPCP(net, testloader):
    
    dataiter = iter(testloader)
    rgb_img, edge_img, gt_pc = next(dataiter)
    # print(rgb_img)
    rgb_img = rgb_img.to(device)
    rg_img_t = rgb_img.clone().detach()
    edge_img = edge_img.to(device)
    gt_pc = gt_pc.to(device)
    output = net(rgb_img, edge_img)

    mi.set_variant("scalar_rgb")
    # print(gt_pc.shape[0])
    for batch_idx in range(gt_pc.shape[0]):

        img = rg_img_t[batch_idx]
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray((img*255).astype(np.uint8), 'RGB')
        img = img.save("Images/%s_%s.png" % ("rgmImg", str(batch_idx)))

        gt_npy = gt_pc[batch_idx].detach().cpu().numpy()
        op_npy = output[batch_idx].detach().cpu().numpy()

        gt_npy = rotatePC(gt_npy, [pi, 0, -pi/2])
        op_npy = rotatePC(op_npy, [pi, 0, -pi/2])

        main(gt_npy, "gt", str(batch_idx))
        main(op_npy, "pred", str(batch_idx))

        scene = mi.load_file("XMLs/%s_%s.xml" % ("gt", str(batch_idx)))
        img = mi.render(scene, spp=256)
        mi.util.write_bitmap("Renders/%s_%s.png" % ("gt", str(batch_idx)), img)

        scene2 = mi.load_file("XMLs/%s_%s.xml" % ("pred", str(batch_idx)))
        img2 = mi.render(scene2, spp=256)
        mi.util.write_bitmap("Renders/%s_%s.png" % ("pred", str(batch_idx)), img2)

        np.save("npy_files/%s_%s.npy" % ("gt",str(batch_idx)), gt_npy)
        np.save("npy_files/%s_%s.npy" % ("pred",str(batch_idx)), op_npy)



        # break


    # img1 = projectImg(gt_pc[0].to("cpu"))
    # img2 = projectImg(output[0].to("cpu"))
    # x, y, z = [ x[0].item() for x in gt_pc[0]], [ x[1].item() for x in gt_pc[0]], [ x[2].item() for x in gt_pc[0]]
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
    # fig.show()
    # x, y, z = [ x[0].item() for x in output[0]], [ x[1].item() for x in output[0]], [ x[2].item() for x in output[0]]
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
    # fig.show()

    # plt.subplot(421)
    # plt.imshow(img1)

    # plt.subplot(422)
    # plt.imshow(img2)

    # rgb_img, edge_img, gt_pc = next(dataiter)
    # rgb_img = rgb_img.to(device)
    # edge_img = edge_img.to(device)
    # gt_pc = gt_pc.to(device)
    # output = net(rgb_img, edge_img)
    # img1 = projectImg(gt_pc[0].to("cpu"))
    # img2 = projectImg(output[0].to("cpu"))
    # x, y, z = [ x[0].item() for x in gt_pc[0]], [ x[1].item() for x in gt_pc[0]], [ x[2].item() for x in gt_pc[0]]
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
    # fig.show()
    # x, y, z = [ x[0].item() for x in output[0]], [ x[1].item() for x in output[0]], [ x[2].item() for x in output[0]]
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
    # fig.show()

    # plt.subplot(423)
    # plt.imshow(img1)

    # plt.subplot(424)
    # plt.imshow(img2)

    # rgb_img, edge_img, gt_pc = next(dataiter)
    # rgb_img = rgb_img.to(device)
    # edge_img = edge_img.to(device)
    # gt_pc = gt_pc.to(device)
    # output = net(rgb_img, edge_img)
    # img1 = projectImg(gt_pc[0].to("cpu"))
    # img2 = projectImg(output[0].to("cpu"))
    # x, y, z = [ x[0].item() for x in gt_pc[0]], [ x[1].item() for x in gt_pc[0]], [ x[2].item() for x in gt_pc[0]]
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
    # fig.show()
    # x, y, z = [ x[0].item() for x in output[0]], [ x[1].item() for x in output[0]], [ x[2].item() for x in output[0]]
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
    # fig.show()

    # plt.subplot(425)
    # plt.imshow(img1)

    # plt.subplot(426)
    # plt.imshow(img2)

    # rgb_img, edge_img, gt_pc = next(dataiter)
    # rgb_img = rgb_img.to(device)
    # edge_img = edge_img.to(device)
    # gt_pc = gt_pc.to(device)
    # output = net(rgb_img, edge_img)
    # img1 = projectImg(gt_pc[0].to("cpu"))
    # img2 = projectImg(output[0].to("cpu"))
    # x, y, z = [ x[0].item() for x in gt_pc[0]], [ x[1].item() for x in gt_pc[0]], [ x[2].item() for x in gt_pc[0]]
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
    # fig.show()
    # x, y, z = [ x[0].item() for x in output[0]], [ x[1].item() for x in output[0]], [ x[2].item() for x in output[0]]
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=0.5))])
    # fig.show()

    # plt.subplot(427)
    # plt.imshow(img1)

    # plt.subplot(428)
    # plt.imshow(img2)


    # plt.show()


# start Traning!
if __name__ == "__main__":

    print("\n")
    print("------- Testing PCP -------")
    print("\n")

    # Setup Hardware:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # if device != "cuda:0":
    logging.info(f"We are training on {device}\n")

    # Setup Training Parameters:
    transform = tf.Compose( [tf.Resize((128,128)),
                             tf.ToTensor()
                            #  tf.Normalize((0.5), (0.5))
                             ])
    batch_size = 12
    start_epoch = 0
    end_epoch = 40
    lr = 0.0005

    # logging.info(f"Loading Train Dataset dataset...\n")
    # img_,mod_,ang_ = parseTrainData()
    # trainset = DatasetLoader(model_paths = mod_, image_paths = img_, angel_paths = ang_, sourceTransform = transform)
    # trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    print("\n"*2)
    logging.info(f"Loading Test Dataset dataset...\n")
    img_,mod_,ang_ = parseValData()
    testset = DatasetLoader(model_paths = mod_, image_paths = img_, angel_paths = ang_, sourceTransform = transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)


    # Get Network:
    logging.info(f"Loading Point Cloud Pyramid...\n")
    net = pointCloudGenerator(MainBranch, AuxiliaryBranch, PCP)
    weightsPCP = torch.load("Pretrained_Networks/PCP.pth")
    net.load_state_dict(weightsPCP["model_state_dict"])


    ## Parameter Freezing
    logging.info(f"Freezing All Parameters...")
    for x in net.parameters():
        x.requires_grad = False

    print("\n")
    # logging.info(f"Starting Training...")
    # logging.info(f"Start Epoch = {start_epoch}, End Epoch = {end_epoch}:\n")
    with torch.no_grad():
        testingPCP(net = net, testloader = testloader)
