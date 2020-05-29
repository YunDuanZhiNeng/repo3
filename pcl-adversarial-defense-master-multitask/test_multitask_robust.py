"""
Created on Sun Mar 24 17:51:08 2019

@author: aamir-mustafa
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_model_multitask import *  # Imports the ResNet Model
import argparse

"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""
parser = argparse.ArgumentParser("Prototype Conformity Loss Implementation")
parser.add_argument('--epsilon', type=float, default=0.03, help="epsilon")  ##
parser.add_argument('--scale', type=float, default=1, help="epsilon")  ##
parser.add_argument('--attack', type=str, default='fgsm')
parser.add_argument('--file-name', type=str, default='Models_Softmax/CIFAR10_Softmax.pth.tar')
parser.add_argument('--outputs-name', type=str, default='outputs')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

num_classes = 10

model = frize_resnet(num_classes=num_classes, depth=110)
if True:
    model = nn.DataParallel(model).cuda()

# Loading Trained Model
softmax_filename = args.file_name
# = 'Models_PCL/CIFAR10_PCL.pth.tar'
# robust_model = 'robust_model.pth.tar'
# checkpoint = torch.load(robust_model)
checkpoint = torch.load(softmax_filename)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Loading Test Data (Un-normalized)
transform_test = transforms.Compose([transforms.ToTensor(), ])

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                       download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, pin_memory=True,
                                          shuffle=False, num_workers=8)

# Mean and Standard Deiation of the Dataset
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t


# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters, out_adv_18=None):
    global out
    adv = img.detach()
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

        noise = 0
    yyr_index=model(normalize(img.clone().detach()))[-1].argmax(dim=-1) == label
    for j in range(iterations):
        f128, f256_15, f256_16, f256_17, f256_18, f256, f1024, \
        out_adv_128, out_adv_256_1, out_adv_256_2, out_adv_256_3, out_adv_256_4, out_adv_256_5, out_adv_256_6, \
        out_adv_256_7, out_adv_256_8, out_adv_256_9, out_adv_256_10, out_adv_256_11, out_adv_256_12, out_adv_256_13, \
        out_adv_256_14, out_adv_256_15, out_adv_256_16, out_adv_256_17, out_adv_256_18, out_adv = model(
            normalize(adv.clone()))
        if args.outputs_name == 'outputs':
            out = out_adv
        elif args.outputs_name == 'outputs128':
            out = out_adv_128
        elif args.outputs_name == 'outputs256_1':
            out = out_adv_256_1
        elif args.outputs_name == 'outputs256_2':
            out = out_adv_256_2
        elif args.outputs_name == 'outputs256_3':
            out = out_adv_256_3
        elif args.outputs_name == 'outputs256_4':
            out = out_adv_256_4
        elif args.outputs_name == 'outputs256_5':
            out = out_adv_256_5
        elif args.outputs_name == 'outputs256_6':
            out = out_adv_256_6
        elif args.outputs_name == 'outputs256_7':
            out = out_adv_256_7
        elif args.outputs_name == 'outputs256_8':
            out = out_adv_256_8
        elif args.outputs_name == 'outputs256_9':
            out = out_adv_256_9
        elif args.outputs_name == 'outputs256_10':
            out = out_adv_256_10
        elif args.outputs_name == 'outputs256_11':
            out = out_adv_256_11
        elif args.outputs_name == 'outputs256_12':
            out = out_adv_256_12
        elif args.outputs_name == 'outputs256_13':
            out = out_adv_256_13
        elif args.outputs_name == 'outputs256_14':
            out = out_adv_256_14
        elif args.outputs_name == 'outputs256_15':
            out = out_adv_256_15
        elif args.outputs_name == 'outputs256_16':
            out = out_adv_256_16
        elif args.outputs_name == 'outputs256_17':
            out = out_adv_256_17
        elif args.outputs_name == 'outputs256_18':
            out = out_adv_256_18
        
        maxk=max((1,2))
        confidence=0
        pred_val, pred_id = model(normalize(adv.clone().detach()))[-1].topk(maxk,1,True,True)
        confidence = pred_val[:,0]-pred_val[:,1]
        loss = criterion(torch.div(out,args.scale),label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad

        # Optimization step
        # print('noise.size: ',noise.size())
        mask= yyr_index.to(dtype=torch.float)
        mask =torch.unsqueeze(torch.unsqueeze( torch.unsqueeze(mask,-1),-1),-1)

        adv.data = adv.data + step *mask* noise.sign()
        # adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach() 


# Loss Criteria
criterion = nn.CrossEntropyLoss()
adv_acc = 0
clean_acc = 0

adv_acc_128 = 0
clean_acc_128 = 0

adv_acc_256_1 = 0
clean_acc_256_1 = 0

adv_acc_256_2 = 0
clean_acc_256_2 = 0

adv_acc_256_3 = 0
clean_acc_256_3 = 0

adv_acc_256_4 = 0
clean_acc_256_4 = 0

adv_acc_256_5 = 0
clean_acc_256_5 = 0

adv_acc_256_6 = 0
clean_acc_256_6 = 0

adv_acc_256_7 = 0
clean_acc_256_7 = 0

adv_acc_256_8 = 0
clean_acc_256_8 = 0

adv_acc_256_9 = 0
clean_acc_256_9 = 0

adv_acc_256_10 = 0
clean_acc_256_10 = 0

adv_acc_256_11 = 0
clean_acc_256_11 = 0

adv_acc_256_12 = 0
clean_acc_256_12 = 0

adv_acc_256_13 = 0
clean_acc_256_13 = 0

adv_acc_256_14 = 0
clean_acc_256_14 = 0

adv_acc_256_15 = 0
clean_acc_256_15 = 0

adv_acc_256_16 = 0
clean_acc_256_16 = 0

adv_acc_256_17 = 0
clean_acc_256_17 = 0

adv_acc_256_18 = 0
clean_acc_256_18 = 0

confidence = 0


eps = args.epsilon  # 8 / 255  # Epsilon for Adversarial Attack

for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)

    clean_acc_128 += 0# torch.sum(model(normalize(img.clone().detach()))[0].argmax(dim=-1) == label).item()
    clean_acc_256_1 += 0 #torch.sum(model(normalize(img.clone().detach()))[1].argmax(dim=-1) == label).item()
    clean_acc_256_2 += 0 #torch.sum(model(normalize(img.clone().detach()))[2].argmax(dim=-1) == label).item()
    clean_acc_256_3 += 0 #torch.sum(model(normalize(img.clone().detach()))[3].argmax(dim=-1) == label).item()
    clean_acc_256_4 += 0 #torch.sum(model(normalize(img.clone().detach()))[4].argmax(dim=-1) == label).item()
    clean_acc_256_5 += 0 #torch.sum(model(normalize(img.clone().detach()))[5].argmax(dim=-1) == label).item()
    clean_acc_256_6 += 0 #torch.sum(model(normalize(img.clone().detach()))[6].argmax(dim=-1) == label).item()
    clean_acc_256_7 += 0 #torch.sum(model(normalize(img.clone().detach()))[7].argmax(dim=-1) == label).item()
    clean_acc_256_8 += 0 #torch.sum(model(normalize(img.clone().detach()))[8].argmax(dim=-1) == label).item()
    clean_acc_256_9 += 0 #torch.sum(model(normalize(img.clone().detach()))[9].argmax(dim=-1) == label).item()
    clean_acc_256_10 += 0 #torch.sum(model(normalize(img.clone().detach()))[10].argmax(dim=-1) == label).item()
    clean_acc_256_11 += 0 #torch.sum(model(normalize(img.clone().detach()))[11].argmax(dim=-1) == label).item()
    clean_acc_256_12 += 0#torch.sum(model(normalize(img.clone().detach()))[12].argmax(dim=-1) == label).item()
    clean_acc_256_13 += 0#torch.sum(model(normalize(img.clone().detach()))[13].argmax(dim=-1) == label).item()
    clean_acc_256_14 += 0#torch.sum(model(normalize(img.clone().detach()))[14].argmax(dim=-1) == label).item()
    clean_acc_256_15 += torch.sum(model(normalize(img.clone().detach()))[22].argmax(dim=-1) == label).item()
    clean_acc_256_16 += torch.sum(model(normalize(img.clone().detach()))[23].argmax(dim=-1) == label).item()
    clean_acc_256_17 += torch.sum(model(normalize(img.clone().detach()))[24].argmax(dim=-1) == label).item()
    clean_acc_256_18 += torch.sum(model(normalize(img.clone().detach()))[25].argmax(dim=-1) == label).item()
    clean_acc += torch.sum(model(normalize(img.clone().detach()))[26].argmax(dim=-1) == label).item()
    
    yyr_index=model(normalize(img.clone().detach()))[26].argmax(dim=-1) == label
    #if yyr_index:
    #    print('need to attack')
    #else:
    #    print('do not need to attack')
    #confidence += torch.sum(model(normalize(img.clone().detach()))[19].argmax(dim=0))
    if True:
        adv = attack(model, criterion, img, label, eps=eps, attack_type=args.attack, iters=10)

        adv_acc_128 += 0 #torch.sum(model(normalize(adv.clone().detach()))[0].argmax(dim=-1) == label).item()
        adv_acc_256_1 += 0 #torch.sum(model(normalize(adv.clone().detach()))[1].argmax(dim=-1) == label).item()
        adv_acc_256_2 += 0 #torch.sum(model(normalize(adv.clone().detach()))[2].argmax(dim=-1) == label).item()
        adv_acc_256_3 += 0 #torch.sum(model(normalize(adv.clone().detach()))[3].argmax(dim=-1) == label).item()
        adv_acc_256_4 += 0 #torch.sum(model(normalize(adv.clone().detach()))[4].argmax(dim=-1) == label).item()
        adv_acc_256_5 += 0 #torch.sum(model(normalize(adv.clone().detach()))[5].argmax(dim=-1) == label).item()
        adv_acc_256_6 += 0 #torch.sum(model(normalize(adv.clone().detach()))[6].argmax(dim=-1) == label).item()
        adv_acc_256_7 += 0 #torch.sum(model(normalize(adv.clone().detach()))[7].argmax(dim=-1) == label).item()
        adv_acc_256_8 += 0 #torch.sum(model(normalize(adv.clone().detach()))[8].argmax(dim=-1) == label).item()
        adv_acc_256_9 += 0 #torch.sum(model(normalize(adv.clone().detach()))[9].argmax(dim=-1) == label).item()
        adv_acc_256_10 += 0 #torch.sum(model(normalize(adv.clone().detach()))[10].argmax(dim=-1) == label).item()
        adv_acc_256_11 += 0 #torch.sum(model(normalize(adv.clone().detach()))[11].argmax(dim=-1) == label).item()
        adv_acc_256_12 += 0#torch.sum(model(normalize(adv.clone().detach()))[12].argmax(dim=-1) == label).item()
        adv_acc_256_13 += 0#torch.sum(model(normalize(adv.clone().detach()))[13].argmax(dim=-1) == label).item()
        adv_acc_256_14 += 0#torch.sum(model(normalize(adv.clone().detach()))[14].argmax(dim=-1) == label).item()
        adv_acc_256_15 += torch.sum(model(normalize(adv.clone().detach()))[22].argmax(dim=-1) == label).item()
        adv_acc_256_16 += torch.sum(model(normalize(adv.clone().detach()))[23].argmax(dim=-1) == label).item()
        adv_acc_256_17 += torch.sum(model(normalize(adv.clone().detach()))[24].argmax(dim=-1) == label).item()
        adv_acc_256_18 += torch.sum(model(normalize(adv.clone().detach()))[25].argmax(dim=-1) == label).item()
        adv_acc += torch.sum(model(normalize(adv.clone().detach()))[26].argmax(dim=-1) == label).item()
        right=torch.sum(model(normalize(adv.clone().detach()))[26].argmax(dim=-1) == label).item()
        #print('right: ',right)
print('scale: ',args.scale, 'attack :',args.attack)
#print('confidence : ', iconfidence)
# print('Batch: {0}'.format(i))
print('{0}\tepsilon :{1:.3%}\n \
        acc :{2:.3%}\t acc 256_18 :{3:.3%}\t acc 256_17 :{4:.3%}\t  acc 256_16 :{5:.3%}\n '
      'acc 256_15 :{6:.3%}\t  acc 256_14 :{7:.3%}\t acc 256_13 :{8:.3%}\t  acc 256_12 :{9:.3%}\n '
      'acc 256_11 :{10:.3%}\t  acc 256_10 :{11:.3%}\t acc 256_9 :{12:.3%}\t  acc 256_8 :{13:.3%}\n '
      'acc 256_7 :{14:.3%}\t  acc 256_6 :{15:.3%}\t acc 256_5 :{16:.3%}\t  acc 256_4 :{17:.3%}\n '
      'acc 256_3 :{18:.3%}\t  acc 256_2 :{19:.3%}\t acc 256_1 :{20:.3%}\t  acc 128:{21:.3%}\n \
          Adv :{22:.3%}\t    Adv 256_18:{23:.3%}\t   Adv 256_17 :{24:.3%}\t     Adv 256_16 :{25:.3%}\n '
      'Adv 256_15 :{26:.3%}\t     Adv 256_14 :{27:.3%}\t Adv 256_13 :{28:.3%}\t     Adv 256_12 :{29:.3%}\n'
      ' Adv 256_11 :{30:.3%}\t     Adv 256_10 :{31:.3%}\t Adv 256_9 :{32:.3%}\t     Adv 256_8 :{33:.3%}\n '
      'Adv 256_7 :{34:.3%}\t     Adv 256_6 :{35:.3%}\t Adv 256_5 :{36:.3%}\t     Adv 256_4 :{37:.3%}\n '
      'Adv 256_3 :{38:.3%}\t     Adv 256_2 :{39:.3%}\t  Adv 256_1 :{40:.3%}\t  Adv 128:{41:.3%}  \
        '.format(args.attack, args.epsilon, clean_acc / len(testset),
                 clean_acc_256_18 / len(testset), clean_acc_256_17 / len(testset), clean_acc_256_16 / len(testset),
                 clean_acc_256_15 / len(testset), clean_acc_256_14 / len(testset), clean_acc_256_13 / len(testset),
                 clean_acc_256_12 / len(testset), clean_acc_256_11 / len(testset), clean_acc_256_10 / len(testset),
                 clean_acc_256_9 / len(testset), clean_acc_256_8 / len(testset), clean_acc_256_7 / len(testset),
                 clean_acc_256_6 / len(testset), clean_acc_256_5 / len(testset), clean_acc_256_4 / len(testset),
                 clean_acc_256_3 / len(testset), clean_acc_256_2 / len(testset), clean_acc_256_1 / len(testset),
                 clean_acc_128 / len(testset), adv_acc / len(testset), adv_acc_256_18 / len(testset),
                 adv_acc_256_17 / len(testset), adv_acc_256_16 / len(testset), adv_acc_256_15 / len(testset),
                 adv_acc_256_14 / len(testset), adv_acc_256_13 / len(testset), adv_acc_256_12 / len(testset),
                 adv_acc_256_11 / len(testset), adv_acc_256_10 / len(testset), adv_acc_256_9 / len(testset),
                 adv_acc_256_8 / len(testset), adv_acc_256_7 / len(testset), adv_acc_256_6 / len(testset),
                 adv_acc_256_5 / len(testset), adv_acc_256_4 / len(testset), adv_acc_256_3 / len(testset),
                 adv_acc_256_2 / len(testset), adv_acc_256_1 / len(testset), adv_acc_128 / len(testset)))
