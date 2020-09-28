from __future__ import print_function

import numpy as np
import os
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import torch.nn.functional as F
from torch.autograd import Variable
from advertorch.attacks import L2PGDAttack, PGDAttack
from keras.utils import to_categorical
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import vgg3 as attn
from scipy.misc import imsave
from dic import B_lable,B_name_set
from cv2 import imwrite
device = 'cuda'

parser = argparse.ArgumentParser(
    description='PGD Adversarial Attacks using GuidedComplementEntropy')
parser.add_argument('--alpha', '-a', default=0.333, type=float,
                    help='alpha for guiding factor')
parser.add_argument('--model',
                    default='./model/origin.tar',
                    type=str,
                    help='load a training model from your (physical) path')
parser.add_argument('--auto_input',
                    default='',
                    type=str,
                    help='load the given image')
parser.add_argument('--L2PGD', action='store_true',
                    help='Using PGD')
parser.add_argument('--concat', action='store_true',
                    help='concat blocks')
parser.add_argument('--id', default=0,
                    type=int, help='store the related results')
parser.add_argument('--brand-size', '-brand_size', default=14,
                    type=int, help='brand size (default: 14)')
parser.add_argument('--model-size', '-model_size', default=19,
                    type=int, help='model size (default: 19)')
parser.add_argument('--batch-size', '-b', default=64,
                    type=int, help='mini-batch size (default: 64)')
parser.add_argument('--eps', '-e', default=50., type=float,
                    help='Set an eplison value for PGD adversarial attacks')
parser.add_argument('--print-freq', '-p', default=40, type=int,
                    metavar='N', help='print frequency (default: 20)')
args = parser.parse_args()
net = attn.__dict__['vgg']()

path = "./attack_result/%d/"%(args.id)
if not os.path.exists(path):
    os.makedirs(path+"/src")
    os.makedirs(path+"/adv")
class cross_entropy_loss(object):
    def __init__(self):
        super(cross_entropy_loss, self).__init__()

    # convert out to softmax probability
    def __call__(self, input, label):
        prob = torch.clamp(3 * torch.softmax(input, 1), 1e-10, 3.0)
        loss = torch.sum(-label* torch.log(prob + 1e-8))
        return loss


def accuracy(data1, data2, value):
    temp1 = MaxNum(data1, value);
    temp2 = MaxNum(data2, value);
    return np.mean(acc(temp1, temp2))


def MaxNum(nums, value):
    temp1 = []
    batch_size = nums.size()[0]
    nums = list(nums)
    for i in range(batch_size):
        temp = []
        Inf = 0
        nt = list(nums[i])
        for t in range(value):
            temp.append(nt.index(max(nt)))
            nt[nt.index(max(nt))] = Inf
        temp.sort()
        temp1.append(temp)
    return temp1


def acc(temp, index):
    accuracy = []
    for k in range(len(temp)):
        accuracy.append((temp[k] == index[k]))
    return accuracy

def get_Top_prob(cal_prob,Total):
    """
    get other top probs based on the Top1
    """
    cal_prob[np.argmax(cal_prob)]=0
    Top_prop = max(cal_prob)/Total
    Attack_Top_label = np.argmax(cal_prob)
    if Attack_Top_label in B_lable:
        AB_category = B_name_set[np.argmax(np.argmax(cal_prob) == B_lable)]
    else:
        AB_category = "None"
    return Top_prop,AB_category,Attack_Top_label


def init(pack):
    inputt,labelt,mlabelt,blabelt = pack
    inputt = torch.FloatTensor(inputt)
    labelt = torch.FloatTensor(labelt)
    mlabelt = torch.FloatTensor(mlabelt)
    blabelt = torch.FloatTensor(blabelt)
    inputt = inputt.cuda()
    labelt = labelt.cuda()
    mlabelt = mlabelt.cuda()
    blabelt = blabelt.cuda()
    return inputt,labelt,mlabelt,blabelt

def save_image(index,input,adv = False):
    input = np.array(input.cpu())
    b,c,h,w = input.shape
    max_len = len(input)
    if not adv:
        for i in range(max_len):
            imwrite("./attack_result/%d/src/%d_%d.png"%(args.id,index,i),input[i].reshape(h,w,c))
    else:
        for i in range(max_len):
            imwrite("./attack_result/%d/adv/%d_%d.png"%(args.id,index,i),input[i].reshape(h,w,c))
def reconstruct_image(index,input,adv = False):
    input = np.array(input.cpu())
    b,c,h,w = input.shape
    sqb = int(np.sqrt(b))
    new_image = np.zeros([sqb*h,sqb*w,c])
    for i in range(sqb):
      for j in range(sqb):
          new_image[i*h:(i+1)*h,j*w:(j+1)*w,:] = input[j*sqb+i].reshape(h,w,c)
    if not adv:
        imwrite("./attack_result/%d/src/%d.png"%(args.id,index),new_image)
    else:
        imwrite("./attack_result/%d/adv/%d.png"%(args.id,index),new_image)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


net.cuda()


print('==> Resuming from checkpoint..')

checkpoint = torch.load(args.model)
net.load_state_dict(checkpoint['state_dict'])
csl = cross_entropy_loss()
print('==> Resuming down!')
cudnn.benchmark = True

"""
attack method
"""
if args.L2PGD:
        adversary = L2PGDAttack(net, loss_fn=csl, eps=args.eps, nb_iter=10, eps_iter=25.,rand_init=True, clip_min=0.0, clip_max=255., targeted=False)
else:
        adversary = PGDAttack(net, loss_fn=csl, eps=args.eps, nb_iter=10, eps_iter=1.75, rand_init=False, clip_min=0.0,clip_max=255., targeted=False)



net.eval()

correct = 0
total = 0


from dataload_test import load_test_list, get_test
test_num = load_test_list()
iters = 100

print(iters)
prec1 = AverageMeter()
prec2 = AverageMeter()
prec3 = AverageMeter()
noise_avg = 0

import json
with open("./attack_result/%d/L_norm.json"%(args.id),"w") as f:
  for iter in range(iters):
    Flag = 0
    inputt, labelt, mlabelt, blabelt = init(get_test(args.batch_size))
    #save_image(iter,inputt)
    if args.concat:
        reconstruct_image(iter,inputt)
    else:
        save_image(iter,inputt)
    #this section is similar to the following section
    out1,out2,out3 = net(inputt)
    cal_prob = np.zeros(args.brand_size)
    for i in range(args.batch_size):
        cal_prob[np.argmax(np.array(out1.data.cpu()),1)[i]] += 1
    Ori_attack_Top1_lable = np.argmax(cal_prob)
    if Ori_attack_Top1_lable in B_lable:
        Ori_AB1_category = B_name_set[np.argmax(np.argmax(cal_prob) == B_lable)]
        Total = np.sum(cal_prob)
        Ori_Top1_Prop = max(cal_prob)/Total
        Ori_Top2_Prop,Ori_AB2_category,Ori_attack_Top2_lable = get_Top_prob(cal_prob,Total)
        Ori_Top3_Prop,Ori_AB3_category,Ori_attack_Top3_lable = get_Top_prob(cal_prob,Total)

    
    adv_inputs = adversary.perturb(inputt, labelt)  # inputs
    if args.concat:
        reconstruct_image(iter,adv_inputs,adv = True)
    else:
        save_image(iter,adv_inputs,adv = True)

    #adv_inputs = inputt
    #this section is similar to the above section
    outt, outt2, outt3 = net(adv_inputs)
    cal_prob = np.zeros(args.brand_size)
    for i in range(args.batch_size):
        cal_prob[np.argmax(np.array(outt.data.cpu()),1)[i]] += 1
    attack_Top1_lable = np.argmax(cal_prob)
    if attack_Top1_lable in B_lable:
        OB_category = B_name_set[np.argmax(np.argmax(np.array(blabelt.cpu()),1)[0] == B_lable)]
        AB1_category = B_name_set[np.argmax(np.argmax(cal_prob) == B_lable)]
        if OB_category == AB1_category:
            Flag = 0
        Total = np.sum(cal_prob)
        Top1_Prop = max(cal_prob)/Total
        Top2_Prop,AB2_category,attack_Top2_lable = get_Top_prob(cal_prob,Total)
        Top3_Prop,AB3_category,attack_Top3_lable = get_Top_prob(cal_prob,Total)
        Noise_level = float(torch.mean(torch.pow(adv_inputs - inputt,2)))
        json_dict = {
             "adv":[(attack_Top1_lable,AB1_category,Top1_Prop),
             (attack_Top2_lable,AB2_category,Top2_Prop),
             (attack_Top3_lable,AB3_category,Top3_Prop)],
             "L2":Noise_level,
             "as":Flag,
             "src":[(Ori_attack_Top1_lable,Ori_AB1_category,Ori_Top1_Prop),
             (Ori_attack_Top2_lable,Ori_AB2_category,Ori_Top2_Prop),
             (Ori_attack_Top3_lable,Ori_AB3_category,Ori_Top3_Prop)],
             "true_label":np.argmax(np.array(blabelt.cpu()),1)[0],
             "true_name":OB_category
        }
        json.dump(json_dict,f)
        f.flush()
    else:
        print("Untill now, this camera is not available")

    top1 = accuracy(outt.data, blabelt, 1)
    top2 = accuracy(outt2.data, mlabelt, 2)
    
    top3 = accuracy(outt3.data[:,:19], labelt[:,:19], 2)
    prec1.update(top1.item(), inputt.size(0))
    prec2.update(top2.item(), inputt.size(0))
    prec3.update(top3.item(), inputt.size(0))
    print('Epoch: [{0}][{1}/{2}]\t'
          'CAMERA BRAND PROB {top1:.3f}\t\t'
          'CAMERA MODEL PROB {top2:.3f}\t\t'
          'CAMERA DEVICE PROB {top3:.3f}'.format(
        0, iter, iters,
        top1=top1, top2=top2, top3=top3
    ))


# these sections are used to print the average Prob
#print(' * CAMERA BRAND PROB {top1.avg:.3f}'.format(top1=prec1))
#print(' * CAMERA MODEL PROB {top2.avg:.3f}'.format(top2=prec2))
#print(' * CAMERA DEVICE PROB {top3.avg:.3f}'.format(top3=prec3))
#print(' * noise_avg@1 {noise:.3f}'.format(noise=noise_avg / iters))
