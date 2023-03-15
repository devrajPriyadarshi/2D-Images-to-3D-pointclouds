import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

def validator(testloader,net):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            img, gt_pc = data
            img = img.to(device)
            gt_pc = gt_pc.to(device)

            pred_pc = net(img)

            ## Define Validator metric

            # total += gt.size(0)
            # correct += (predicted == labels).sum().item()

    print(f'\nAccuracy of the network on the 10000 test images: {100 * correct // total} %\n')

    return correct/total