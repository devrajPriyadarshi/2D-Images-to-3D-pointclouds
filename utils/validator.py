import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

def validator(testloader, net, criterion):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        running_loss = 0
        # i = 0
        for data in testloader:
            # i+=1
            # print(i)
            # if i>10:
            #     break

            rgb_img, edge_img, gt_pc = data

            rgb_img = rgb_img.to(device)
            edge_img = edge_img.to(device)
            gt_pc = gt_pc.to(device)


            output = net(rgb_img, edge_img)
            loss = criterion(output, gt_pc, a=0, b=1, c=0)

            running_loss+=loss.item()

    print("Total chamfer accuracy: ", running_loss)

    return running_loss