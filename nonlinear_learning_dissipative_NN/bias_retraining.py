# -*- coding: utf-8 -*-
import os
import argparse
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from mat4py import savemat

import sys
sys.argv=['']
del sys

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri8','dopri5', 'adams'], default='dopri8')
parser.add_argument('--data_size', type=int, default=10000)
parser.add_argument('--batch_time', type=int, default=6000)   # num of points collected in an interval for a starting point in a batch
parser.add_argument('--batch_size', type=int, default=500)  # num of starting point in a batch
parser.add_argument('--niters_max', type=int, default=4000) 
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--viz', default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--traj_num', type=int, default=1)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



## For biases retraining 
np.random.seed(456)
# Starting points
true_y0s = []


states01 = np.array([[0.1,0.1]])
# step response
true_y01 = torch.tensor(np.float32(np.append(states01, [[1,0]], axis=1))).to(device)
true_y0s += [true_y01]

t = torch.linspace(0., 15., args.data_size).to(device)


class Lambda(nn.Module):
    def forward(self, t, y):        
        # mass-spring-damper system with m=k=c=alpha=1
        x1 = y[:,0]
        x2 = y[:,1]
        u = y[:,2]
        t = t.cpu()
        a = torch.tensor([x2]).to(device)
        b = torch.tensor([-x2-x1-x1**3+u]).to(device)
        c = torch.tensor([0]).to(device)
        d = torch.tensor([0]).to(device)
        result = torch.cat((a,b,c,d),0)
        
        # print(result)
        
        
        return result



true_ys = []
with torch.no_grad():
    for i in range(args.traj_num):
        true_y0 = true_y0s[i]
        true_y = odeint(Lambda(), true_y0, t, method='dopri8')  # get training data (ground truth)
        true_ys += [true_y]


# Check if all the trajectories are stablizing
for i in range(args.traj_num):
    true_y = true_ys[i]
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131)
    ax_traj.cla()
    ax_traj.set_title('Trajectories')

    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x1,x2')

    ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], 'b-', t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')

    makedirs("Results/Survey_MSD/figs")
    plt.savefig("Results/Survey_MSD/figs/true_traj_time_"+str(i+1)+".png")



def get_batch():
    ind = np.random.choice(a=args.traj_num, size=1, replace=False)[0]
    
    # For training
    true_y_train = true_ys[ind]
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time , dtype=np.int64), args.batch_size, replace=False))
    # randomly pick a starting point
    batch_y0_train = true_y_train[s]  # (M, D)
    batch_t_train = t[:args.batch_time]  # (T)
    # noise added to samples
    batch_y_train = torch.stack([true_y_train[s + i] + (torch.rand([1,4])*0.01).to(device)
          for i in range(args.batch_time)], dim=0)  # (T, M, D)
    
    # For validation, set to be under the same trajectory
    true_y_val = true_ys[ind]
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time , dtype=np.int64), args.batch_size, replace=False))
    # randomly pick a starting point
    batch_y0_val = true_y_val[s]  # (M, D)
    batch_t_val = t[:args.batch_time]  # (T)
    # noise added to samples
    batch_y_val = torch.stack([true_y_val[s + i] + (torch.randn([1,4])*0.01).to(device)
          for i in range(args.batch_time)], dim=0)  # (T, M, D)
    
    
    return [batch_y0_train.to(device), batch_t_train.to(device), batch_y_train.to(device), ind], [batch_y0_val.to(device), batch_t_val.to(device), batch_y_val.to(device), ind]



def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:

        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
       
        ax_traj.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(pred_y.cpu().detach().numpy()[:, 0, 0], pred_y.detach().cpu().numpy()[:, 0, 1], 'b--')

        makedirs("Results/Survey_MSD/figs")

        ## For biases retraining
        plt.savefig('Results/Survey_MSD/figs/biases_retrainig_itr'+str(itr)+'_4164_'+'actv01_nolast'+'.png')



class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 4),
        )





        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)



    def forward(self, t, y):
        return self.net(y)   #y**3




# get weights
import scipy.io
data = scipy.io.loadmat('Results/Survey_MSD/weights_adjusted.mat')


weights = {}
# Always check about the rolling and unrolling!
# Survey MSD version
weights['layer1'] = np.float32(data['W1_adj'].reshape(data['NN_dim'][0,0], data['NN_dim'][0,1])).T
weights['layer2'] = np.float32(data['W2_adj'].reshape(data['NN_dim'][0,1], data['NN_dim'][0,2])).T







if __name__ == '__main__':
    
    # Set the parameter for the Leaky ReLU function
    func = ODEFunc().to(device)


    ## For biases retraining only
    # load and fix weights
    l = 0
    for m in func.net.modules():
        if isinstance(m, nn.Linear):
          l += 1
          m.weight = torch.nn.Parameter(torch.tensor(weights['layer'+str(l)]).to(device))
          m.weight.requires_grad = False
          # print(m.weight.dtype)
          # print(m.bias.dtype)

    for name, param in func.named_parameters():
        if param.requires_grad:
            print('Trainable',name)
    
    
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-2)


    itr = 0
    # gather the process of training
    pred_ys = []
    loss = 10
    loss_vals = []


    while (loss>1e-4)  and (itr < args.niters_max):
        itr += 1
        optimizer.zero_grad()
        train_data, val_data = get_batch()
        
        # training
        batch_y0, batch_t, batch_y, ind = train_data
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()


       
        if itr % args.test_freq == 0:
            print('Iter {:04d} | Current Training Loss {:.6f}'.format(itr, loss.item()))
            if itr >= 0:
                with torch.no_grad():
                    # validation
                    batch_y0, batch_t, batch_y, ind = val_data
                    pred_y = odeint(func, batch_y0, batch_t).to(device)
                    loss_val = torch.mean(torch.abs(pred_y - batch_y))
                    loss_vals += [loss_val]
                    print('Iter {:04d} | Current Validation Loss {:.6f}'.format(itr, loss_val.item()))
                    



        if itr % args.test_freq == 0:
            with torch.no_grad():
                # predict and visualize on the start same as validation
                true_y0 = true_y0s[ind]
                true_y = true_ys[ind]
                pred_y = odeint(func, true_y0, t, method="dopri8")
                # pred_ys += [pred_y]
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Prediction Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, itr)
        

        
        if itr % args.test_freq == 0:
            weights = {}  # layer num: flattened weights
            biases = {} # layer num:  biases
            for l in range(int((len(func.net)+1)/2)):
                weights['W'+str(l+1)] = func.net[2*l].weight.cpu().data.reshape(-1,1).tolist()
                biases['b'+str(l+1)] = func.net[2*l].bias.cpu().data.reshape(-1,1).tolist()
        

        
            # For saving retrained bias
            savemat('Results/Survey_MSD/retrained_biases_4164_'+str(itr)+'_actv01_nolast'+'.mat', biases)
                
               
    
   
