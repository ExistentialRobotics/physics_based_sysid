import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from numpy import linalg as LA

from model import init_NNs
# used for nonlinear power flow simulation.
from scipy.io import loadmat
import pandapower as pp
import gymnasium as gym
import matplotlib.pyplot as plt

# Build the adjecent matrix for IEEE 13-bus sytem (except the root bus):
A = -np.eye(12)
A[1,0]=1
A[2,0]=1
A[3,0]=1
A[4,1]=1
A[5,2]=1
A[6,2]=1
A[7,2]=1
A[8,3]=1
A[9,5]=1
A[10,5]=1
A[11,7]=1
# how this adjecent matrix is built:
# lines(rows):0-1,1-2,1-5,1-3,2-4,5-7,5-8,5-9,3-6,7-10,7-11,9-12
# buses (columns):1,2,5,3,4,7,8,9,6,10,11,12

# Impedence values for each lines
X = np.diag([0.3856,0.1276,0.3856,0.1119,0.0765,0.0771,0.1928,0.0423,0.1119,0.0766,0.0766,0.0423])

F = -np.linalg.inv(A)
X =2*F@X@F.T
# Thus the linearized power flow equation is
# v = Xq + v_{env}, note here v is squared voltage magniture.

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def create_13bus():
    pp_net = pp.converter.from_mpc('./data/case_13.mat', casename_mpc_file='case_mpc')

    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 2, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 7, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 9, p_mw = 0, q_mvar=0)

    pp.create_sgen(pp_net, 1, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 3, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 4, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 5, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 6, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 8, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 10, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 11, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 12, p_mw = 0, q_mvar=0)

    # In the original IEEE 13 bus system, there is no load in bus 3, 7, 8.
    # Add the load to corresponding node for dimension alignment in RL training
    pp.create_load(pp_net, 3, p_mw = 0, q_mvar=0)
    pp.create_load(pp_net, 7, p_mw = 0, q_mvar=0)
    pp.create_load(pp_net, 8, p_mw = 0, q_mvar=0)

    return pp_net


class IEEE13bus(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1, vmax=1.05, vmin=0.95, all_bus=False):
        self.network =  pp_net
        self.obs_dim = 1
        self.action_dim = 1
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        # if self.agentnum == 12:  # comment out for mpc experiments
        #     all_bus=True
        self.v0 = v0
        self.vmax = vmax
        self.vmin = vmin

        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        self.all_bus = all_bus

        self.state = np.ones(self.agentnum, )

    def step(self, action):
        # state-transition dynamics
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i, 'q_mvar'] = action[i]

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')

        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state

    def reset0(self, seed=1): #reset voltage to nominal value

        self.network.load['p_mw'] = 0*self.load0_p
        self.network.load['q_mvar'] = 0*self.load0_q

        self.network.sgen['p_mw'] = 0*self.gen0_p
        self.network.sgen['q_mvar'] = 0*self.gen0_q

        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state



def main_linear():
    print('Train and test with the linear power flow.')
    ### Data generation
    # Obviously, v = Xq + v_{env} is monotone w.r.t. q
    # For simplicity, we set v_{env} to one.
    q_data = (np.random.rand(12,10000).astype(np.float32) - 0.5)*0.2 #range [-0.1,0.1]
    v_data = X@q_data + 1
    batch_size = 256
    epochs = 300
    v_data = np.float32(v_data)
    dataset = torch.utils.data.TensorDataset(torch.tensor(q_data.T), torch.tensor(v_data.T))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # FTN_net, FTN_optimizer, cvx_net,cvx_optimizer,criterion = init_NNs()
    FTN_net, FTN_optimizer, cvx_net, cvx_optimizer, simple_NN, NN_optimizer, criterion = init_NNs()

    loss_cvx_list = []
    loss_ftn_list = []
    loss_nn_list = []
    for epoch in range(epochs):
        epoch_loss = 0
        cvx_epoch_loss = 0
        nn_epoch_loss = 0
        for i, (q, v) in enumerate(train_dataloader):
            # Reset gradients
            FTN_optimizer.zero_grad()
            cvx_optimizer.zero_grad()
            NN_optimizer.zero_grad()

            # Forward pass: Compute predicted actions by passing states to the model
            predicted_v = FTN_net(q.to(device))
            cvx_predicted_v = cvx_net(q.to(device))
            NN_predicted_v = simple_NN(q.to(device))

            # Calculate loss
            loss = criterion(predicted_v, v.to(device))
            cvx_loss = criterion(cvx_predicted_v, v.to(device))
            nn_loss = criterion(NN_predicted_v, v.to(device))

            # Backward pass: Compute gradient of the loss with respect to model parameters
            loss.backward()
            cvx_loss.backward()
            nn_loss.backward()

            # Perform a single optimization step (parameter update)
            FTN_optimizer.step()
            cvx_optimizer.step()
            NN_optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            cvx_epoch_loss += cvx_loss.item()
            nn_epoch_loss += nn_loss.item()
        loss_cvx_list.append(cvx_epoch_loss)
        loss_ftn_list.append(epoch_loss)
        loss_nn_list.append(nn_epoch_loss)

        # Print average loss for the epoch
        if epoch%10==0:
            print(f"****FTN method****(Linear): Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/i:.5f}")
            print(f"****CVX method****(Linear): Epoch {epoch+1}/{epochs}, Loss: {cvx_epoch_loss/i:.5f}")
            print(f"****NN method****(Linear): Epoch {epoch+1}/{epochs}, Loss: {nn_epoch_loss/i:.5f}")
    ################
    ###TEST
    q_test = (np.random.rand(12,1000).astype(np.float32) - 0.5)*0.2 #range [-0.1,0.1]
    v_test = X@q_test + 1
    
    predicted_v = FTN_net(torch.tensor(q_test.T).to(device))
    cvx_predicted_v = cvx_net(torch.tensor(q_test.T).to(device))
    nn_predicted_v = simple_NN(torch.tensor(q_test.T).to(device))
    # check the mean square error
    print('Error of FTN ', np.mean(np.square(predicted_v.cpu().detach().numpy() - v_test.T)))
    print('Error of CVX ', np.mean(np.square(cvx_predicted_v.cpu().detach().numpy() - v_test.T)))
    print('Error of NN ', np.mean(np.square(nn_predicted_v.cpu().detach().numpy() - v_test.T)))
    
def main_non_linear():
    print('Train and test with the nonlinear simulation.')
    net = create_13bus()
    injection_bus = np.array([1,2,5,3,4,7,8,9,6,10,11,12])
    env = IEEE13bus(net, injection_bus)
    # generate data
    v_list = []
    q_list = []
    env.reset0()
    for i in range(10000):
        q = (np.random.rand(12).astype(np.float32) - 0.5)*0.4 #range [-0.1,0.1]
        env.step(q)
        v_list.append(env.state)
        q_list.append(q)
    v_data = np.array(v_list)
    q_data = np.array(q_list)
    
    batch_size = 128
    epochs = 150
    v_data = np.float32(v_data)
    q_data = np.float32(q_data)
    dataset = torch.utils.data.TensorDataset(torch.tensor(q_data), torch.tensor(v_data))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # FTN_net, FTN_optimizer, cvx_net,cvx_optimizer,criterion = init_NNs()
    FTN_net, FTN_optimizer, cvx_net, cvx_optimizer, simple_NN, NN_optimizer, criterion = init_NNs()

    # relu_mono = Stacked_relu(hidden_dim=100).to(device)
    # relu_mono_optimizer = torch.optim.Adam(relu_mono.parameters(), lr=1e-3)

    FTN_scheduler = torch.optim.lr_scheduler.StepLR(FTN_optimizer, step_size=50, gamma=0.95)

    loss_cvx_list = []
    loss_ftn_list = []
    loss_nn_list = []
    for epoch in range(epochs):
        epoch_loss = 0
        cvx_epoch_loss = 0
        nn_epoch_loss = 0
        relu_epoch_loss = 0
        for i, (q, v) in enumerate(train_dataloader):
            # Reset gradients
            FTN_optimizer.zero_grad()
            cvx_optimizer.zero_grad()
            NN_optimizer.zero_grad()
            # relu_mono_optimizer.zero_grad()

            # Forward pass: Compute predicted actions by passing states to the model
            predicted_v = FTN_net(q.to(device))
            cvx_predicted_v = cvx_net(q.to(device))
            NN_predicted_v = simple_NN(q.to(device))
            # relu_predicted_v = relu_mono(q.to(device))

            # Calculate loss
            loss = criterion(predicted_v, v.to(device))
            cvx_loss = criterion(cvx_predicted_v, v.to(device))
            nn_loss = criterion(NN_predicted_v, v.to(device))
            # relu_loss = criterion(relu_predicted_v, v.to(device))

            # Backward pass: Compute gradient of the loss with respect to model parameters
            loss.backward()
            cvx_loss.backward()
            nn_loss.backward()
            # relu_loss.backward()

            # Perform a single optimization step (parameter update)
            FTN_optimizer.step()
            cvx_optimizer.step()
            NN_optimizer.step()
            # relu_mono_optimizer.step()

            FTN_scheduler.step()

            # Accumulate loss
            epoch_loss += loss.item()
            cvx_epoch_loss += cvx_loss.item()
            nn_epoch_loss += nn_loss.item()
            # relu_epoch_loss += relu_loss.item()
        loss_cvx_list.append(cvx_epoch_loss)
        loss_ftn_list.append(epoch_loss)
        loss_nn_list.append(nn_epoch_loss)

        # Print average loss for the epoch
        if epoch%10==0:
            print(f"****FTN method****(Nonlinear): Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/i:.5f}")
            print(f"****CVX method****(Nonlinear): Epoch {epoch+1}/{epochs}, Loss: {cvx_epoch_loss/i:.5f}")
            print(f"****NN method****(Nonlinear): Epoch {epoch+1}/{epochs}, Loss: {nn_epoch_loss/i:.5f}")
    ##### TEST
    v_list = []
    q_list = []
    env.reset0()
    for i in range(1000):
        q = (np.random.rand(12).astype(np.float32) - 0.5)*0.3 
        env.step(q)
        v_list.append(env.state)
        q_list.append(q)
    v_test = np.array(v_list)
    q_test = np.array(q_list)

    v_test = np.float32(v_test)
    q_test = np.float32(q_test)

    predicted_v = FTN_net(torch.tensor(q_test).to(device))
    cvx_predicted_v = cvx_net(torch.tensor(q_test).to(device))
    nn_predicted_v = simple_NN(torch.tensor(q_test).to(device))
    #### MSE test performance
    print('Error of FTN ', np.mean(np.square(predicted_v.cpu().detach().numpy() - v_test)))
    print('Error of CVX ', np.mean(np.square(cvx_predicted_v.cpu().detach().numpy() - v_test)))
    print('Error of NN ', np.mean(np.square(nn_predicted_v.cpu().detach().numpy() - v_test)))
    return FTN_net, cvx_net, simple_NN
    

def visualize(FTN_net, cvx_net, simple_NN):
    net = create_13bus()
    injection_bus = np.array([1,2,5,3,4,7,8,9,6,10,11,12])
    env = IEEE13bus(net, injection_bus)
    # Load real world data
    q = loadmat('./data/aggr_q.mat')
    q = q['q']
    # Broadcast q to Tx12
    q_broadcasted = np.tile(q, (1, 12)) 
    
    # Generate positive random weights between 0 and 1
    weights = np.random.rand(12)*0.02

    # Initialize lists to store v and q values
    v_traj_true = []

    v_traj_cvx = []
    v_traj_ftn = []
    v_traj_nn = []

    # Reset the environment
    env.reset0()

    # Generate trajectory with random scaling
    for i in range(q.shape[0]):
        q_scaled = q_broadcasted[i] * weights - 0.02  # Scale q values with weights
        env.step(-q_scaled)

        q_scaled = np.float32(q_scaled)

        ftn_v = FTN_net(torch.tensor(-q_scaled).unsqueeze(0).to(device)).cpu().detach().numpy()
        cvx_v = cvx_net(torch.tensor(-q_scaled).unsqueeze(0).to(device)).cpu().detach().numpy()
        nn_v = simple_NN(torch.tensor(-q_scaled).unsqueeze(0).to(device)).cpu().detach().numpy()

        v_traj_cvx.append(cvx_v)
        v_traj_ftn.append(ftn_v)
        v_traj_nn.append(nn_v)

        v_traj_true.append(env.state)

    # Convert lists to arrays
    v_trajectory = np.array(v_traj_true)
    v_traj_cvx = np.array(v_traj_cvx)
    v_traj_ftn = np.array(v_traj_ftn)
    v_traj_nn = np.array(v_traj_nn)

    plt.figure(figsize=(6, 3.5))  # Adjust figure size to fit two-column paper

    id = 5

    # Plot trajectories with compact labels
    plt.plot(v_trajectory[:, id], label='True Data', linewidth=1.5)
    plt.plot(v_traj_cvx[:, id], label='MNN1', linewidth=1.5, linestyle='--')
    plt.plot(v_traj_ftn[:, 0, id], label='MNN2', linewidth=1.5, linestyle='--')
    plt.plot(v_traj_nn[:, 0, id], label='FNN', linewidth=1.5, linestyle=':')

    # Adjust title, labels, and legend
    plt.title('Voltage Trajectory', fontsize=14)
    plt.xlabel('Time (Hours)', fontsize=12)
    plt.ylabel('Voltage (p.u.)', fontsize=12)

    plt.legend(loc='upper right', ncol=2, fontsize=10, frameon=False)

    # Set x-ticks and labels for a day cycle
    plt.xticks(np.arange(0, v_traj_nn.shape[0], 3600), 
            ['00:00', '06:00', '12:00', '18:00', '24:00'], fontsize=10)

    # Add grid for clarity
    plt.grid(alpha=0.5)

    # Save figure with high resolution for publication
    plt.tight_layout()
    plt.savefig('trajectory_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    
    
if __name__ == "__main__":
    main_linear()
    FTN_net, cvx_net, simple_NN = main_non_linear()
    visualize(FTN_net, cvx_net, simple_NN)
    
