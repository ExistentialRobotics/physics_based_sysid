from model import *
from env import *
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib.colors import Normalize

# Consider a simple damped Pendulum, if you are interested, this simulation is capable to simulate a n-link pendulum system by setting N=n.
N=1

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def f_true(x):
    g_l = 9.81
    b = 0.3
    return np.array([x[:,1], -b*x[:,1] + g_l*np.sin(x[:,0]-np.pi)]).T

#helper util functions
def to_variable(X, cuda=True):
    if isinstance(X, (tuple, list)):
        return tuple(to_variable(x) for x in X)
    else:
        X = Variable(X)
        if cuda:
            return X.cuda().requires_grad_()
        return X.requires_grad_()

def runbatch(model, loss, batch, no_proj=False):
    X, Yactual = batch
    X = to_variable(X, cuda=torch.cuda.is_available())
    Yactual = to_variable(Yactual, cuda=torch.cuda.is_available())

    Ypred = model(X)
    return loss(model, Ypred, Yactual, X, no_proj), Ypred

def main():
    # Generate the dataset, X is the input, the labels are the corresponding gradients.
    gfunc = pendulum_gradient(1)
    X_data = (np.random.rand(10000, 2).astype(np.float32) - 0.5) * 2 * np.pi # Pick values in range [-pi, pi] radians, radians/sec

    test = gfunc(X_data)
    ref = f_true(X_data)

    differences = sorted(np.sum((test - ref) ** 2, axis=1))[-8:]
    print(differences)
    assert differences[-1] < 1e-8
    
    # Save the dataset
    np.savez('data.npz',X=X_data,Y=test)
    
    #build simple NN
    model_simple = nn.Sequential(
        nn.Linear(2*N, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 2*N)
    ).to(device)
    loss_ = nn.MSELoss()
    loss_simple = lambda model, Ypred, Yactual, X, no_proj=False, **kw: loss_(Ypred, Yactual)
    model_simple = train_simple(model_simple, X_data, test, loss_simple)
    
    #build the stable NN (by projection)
    fhat = nn.Sequential(
        nn.Linear(2*N, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 2*N)
    ).to(device)
    projfn_eps = 5
    V = MakePSD(ICNN([2*N, 60, 60, 1], activation=ReHU(float(7))), 2*N, eps=projfn_eps, d=1.0).to(device)
    model_stable = Dynamics(fhat, V, alpha=0.001).to(device)
    model_stable = train_w_projection(model_stable, X_data, test)
    
     #build the stable NN (by soft penalty)
    fhat = nn.Sequential(
        nn.Linear(2*N, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 2*N)
    ).to(device)
    projfn_eps = 5
    V = MakePSD(ICNN([2*N, 60, 60, 1], activation=ReHU(float(7))), 2*N, eps=projfn_eps, d=1.0).to(device)
    model_stable_no_proj = Dynamics(fhat, V, alpha=0.01, no_proj=True).to(device)
    model_stable_no_proj = train_w_soft_pen(model_stable_no_proj, X_data, test)
    
    print('####### Training finished ##########')
    simulate(model_simple, model_stable, model_stable_no_proj)
    
def simulate(model_simple, model_stable, model_stable_no_proj):
    ################################################
    ## True simulation
    #timestep
    h = 0.01
    physics = pendulum_gradient(N)

    energy = pendulum_energy(N)
    n = N
    number = 100
    steps = 1000
    X_init = np.zeros((number, 2 * n)).astype(np.float32)
    X_init[:,:] = (np.random.rand(number, 2*n).astype(np.float32) - 0.5) * np.pi/4 # Pick values in range [-pi/8, pi/8] radians, radians/sec
    X_phy = np.zeros((steps, *X_init.shape), dtype=np.float32)
    X_phy[0,...] = X_init
    for i in range(1, steps):
        k1 = h * physics(X_phy[i-1,...])
        k2 = h * physics(X_phy[i-1,...] + k1/2)
        k3 = h * physics(X_phy[i-1,...] + k2/2)
        k4 = h * physics(X_phy[i-1,...] + k3)
        X_phy[i,...] = X_phy[i-1,...] + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        assert not np.any(np.isnan(X_phy[i,...]))
        
    ###################################################
    # Simulated with simple NN
    model_simple.eval()
    X_nn = to_variable(torch.tensor(X_phy[0,:,:]), cuda=torch.cuda.is_available())
    log_y = []
    errors = np.zeros((steps,))
    for i in range(1, steps):
        X_nn.requires_grad = True
        k1 = h * model_simple(X_nn)
        k1 = k1.detach()
        k2 = h * model_simple(X_nn + k1/2)
        k2 = k2.detach()
        k3 = h * model_simple(X_nn + k2/2)
        k3 = k3.detach()
        k4 = h * model_simple(X_nn + k3)
        k4 = k4.detach()
        X_nn = X_nn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        X_nn = X_nn.detach()

        y = X_nn.cpu().numpy()
        log_y.append(y)

        vel_error = np.sum((X_phy[i,:,n:] - y[:,n:])**2)
        ang_error = (X_phy[i,:,:n] - y[:,:n])
        while np.any(ang_error >= np.pi):
            ang_error[ang_error >= np.pi] -= 2*np.pi
        while np.any(ang_error < -np.pi):
            ang_error[ang_error < -np.pi] += 2*np.pi

        ang_error = np.sum(ang_error**2)
        errors[i] = (vel_error + ang_error)
    log_y = np.array(log_y)
    plt.plot(X_phy[:,0,0], label='True Trajectory')
    plt.plot(log_y[:,0,0], label='Simple NN')
    plt.title('Angel Trajectory with simple neural network')
    plt.legend()
    plt.savefig('./figure/angle_trajectory_simple.png')
    plt.close()
    ###################################################
    # Simulated with Stable NN with proj
    model_stable.eval()
    X_nn = to_variable(torch.tensor(X_phy[0,:,:]), cuda=torch.cuda.is_available())
    log_y_stable = []
    errors = np.zeros((steps,))
    for i in range(1, steps):
        X_nn.requires_grad = True
        k1 = h * model_stable(X_nn)
        k1 = k1.detach()
        k2 = h * model_stable(X_nn + k1/2)
        k2 = k2.detach()
        k3 = h * model_stable(X_nn + k2/2)
        k3 = k3.detach()
        k4 = h * model_stable(X_nn + k3)
        k4 = k4.detach()
        X_nn = X_nn + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        X_nn = X_nn.detach()

        y = X_nn.cpu().numpy()
        log_y_stable.append(y)

        vel_error = np.sum((X_phy[i,:,n:] - y[:,n:])**2)
        ang_error = (X_phy[i,:,:n] - y[:,:n])
        while np.any(ang_error >= np.pi):
            ang_error[ang_error >= np.pi] -= 2*np.pi
        while np.any(ang_error < -np.pi):
            ang_error[ang_error < -np.pi] += 2*np.pi

        ang_error = np.sum(ang_error**2)
        errors[i] = (vel_error + ang_error)
    log_y_stable = np.array(log_y_stable)
    plt.plot(X_phy[:,0,0], label='True Trajectory')
    plt.plot(log_y_stable[:,0,0], label='Stable with projection')
    plt.title('Angel Trajectory with stable dynamics')
    plt.legend()
    plt.savefig('./figure/angle_trajectory_stable.png')
    plt.close()
    ###################################################
    # Simulated with Stable NN without proj
    model_stable_no_proj.eval()
    X_nn_no_proj = to_variable(torch.tensor(X_phy[0,:,:]), cuda=torch.cuda.is_available())
    log_y_no_proj = []
    errors_no_proj = np.zeros((steps,))
    for i in range(1, steps):
        X_nn_no_proj.requires_grad = True
        k1 = h * model_stable_no_proj(X_nn_no_proj)
        k1 = k1.detach()
        k2 = h * model_stable_no_proj(X_nn_no_proj + k1/2)
        k2 = k2.detach()
        k3 = h * model_stable_no_proj(X_nn_no_proj + k2/2)
        k3 = k3.detach()
        k4 = h * model_stable_no_proj(X_nn_no_proj + k3)
        k4 = k4.detach()
        X_nn_no_proj = X_nn_no_proj + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        X_nn_no_proj = X_nn_no_proj.detach()

        y_no_proj = X_nn_no_proj.cpu().numpy()
        log_y_no_proj.append(y_no_proj)

        vel_error_no_proj = np.sum((X_phy[i,:,n:] - y_no_proj[:,n:])**2)
        ang_error_no_proj = (X_phy[i,:,:n] - y_no_proj[:,:n])
        while np.any(ang_error_no_proj >= np.pi):
            ang_error_no_proj[ang_error_no_proj >= np.pi] -= 2*np.pi
        while np.any(ang_error_no_proj < -np.pi):
            ang_error_no_proj[ang_error_no_proj < -np.pi] += 2*np.pi

        ang_error_no_proj = np.sum(ang_error_no_proj**2)
        errors_no_proj[i] = (vel_error_no_proj + ang_error_no_proj)
    log_y_no_proj = np.array(log_y_no_proj)
    plt.plot(X_phy[:,0,0], label='True Trajectory')
    plt.plot(log_y_no_proj[:,0,0], label='Stable with soft penalty')
    plt.title('Angel Trajectory with stable dynamics without projection')
    plt.legend()
    plt.savefig('./figure/angle_trajectory_stable_wo_proj.png')
    plt.close()
    
    ###### Lyapunov contour visualization
    X,Y,Z=plot_lyapunov_contour(model_stable, model_stable.V, [-0.5, 0.5], [-1, 1.2])
    id = 0
    plt.plot(log_y_stable[:,id,0],log_y_stable[:,id,1], 'r-', label='Predicted Trajectory')
    plt.plot(X_phy[:,id,0],X_phy[:,id,1], 'y-', label='True Trajectory')
    plt.title('Lyapunov Contour With Projection')
    plt.legend()
    plt.savefig('./figure/lyapunov_contour_proj.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    X,Y,Z=plot_lyapunov_contour(model_stable_no_proj, model_stable_no_proj.V, [-0.5, 0.5], [-1, 1.2])
    id = 0
    log_y_no_proj = np.array(log_y_no_proj)
    plt.plot(log_y_no_proj[:,id,0],log_y_no_proj[:,id,1], 'r-', label='Predicted Trajectory')
    plt.plot(X_phy[:,id,0],X_phy[:,id,1], 'y-', label='True Trajectory')
    plt.title('Lyapunov Contour Without Projection')
    plt.legend()
    plt.savefig('./figure/lyapunov_contour_no_proj.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
def plot_lyapunov_contour(model, lyapunov_function, x_range, y_range):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
   
    Z = np.array([[lyapunov_function(torch.tensor([[xi, yi]], dtype=torch.float32).to(device)).item() for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
    
    #  Normalize Z to the [0, 1] range --- Due to package or other issues, each training may have different V landscapes, to ease the burden, we use normalization for visualization
    # Added a small epsilon to prevent division by zero if Z is constant
    print(np.max(Z), np.min(Z))
    Z = (Z - np.min(Z)) / (np.max(Z) - np.min(Z) + 1e-9)
    xx = np.linspace(x_range[0], x_range[1], 20)
    yy = np.linspace(y_range[0], y_range[1], 20)
    XX, YY = np.meshgrid(xx, yy)
    U, V = np.array([[model.fhat(torch.tensor([[xi, yi]], dtype=torch.float32).to(device)).cpu().detach().numpy() for xi, yi in zip(x_row, y_row)] for x_row, y_row in zip(XX, YY)]).squeeze().transpose(2, 0, 1)

    # Add a thinner color bar
    # Set normalization for contour levels
    # contour_norm = Normalize(vmin=5, vmax=35)
    # Create the plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Slightly larger figure for better color bar alignment
    cf = ax.contourf(X, Y, Z, levels=50, cmap='viridis')

    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, aspect=10)
    # cbar.set_label("Lyapunov Value", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])  # Fewer ticks for clarity
    
    ax.quiver(XX, YY, U, V, color='white')
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$\dot{x}$', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    return X,Y,Z   
    
    
    
    
    

def train_w_projection(model_stable, X_data, test):
    epochs = 1000
    batch_size = 2000
    learning_rate = 0.01
    min_loss = 100000
    # To minimize installation, we do not use summary writter here
    loss_record = []
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_data), torch.tensor(test))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model_stable
    loss = loss_stable

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        loss_parts = []
        model.train()
        for data in train_dataloader:
            optimizer.zero_grad()
            loss, _ = runbatch(model, loss_stable, data)
            loss_parts.append(np.array([l.cpu().item() for l in loss]))

            optim_loss = loss[0] if isinstance(loss, (tuple, list)) else loss
            optim_loss.backward()
            optimizer.step()
            epoch_loss = sum(loss_parts) / len(dataset)

        print(f'Epoch {epoch}, loss: {epoch_loss}')
        loss_record.append(epoch_loss)
        if np.sum(epoch_loss)< min_loss:
            min_loss = np.sum(epoch_loss)
            torch.save(model_stable.state_dict(), 'checkpoint/stable_with_proj.pth')
    return model

def train_w_soft_pen(model_stable_no_proj, X_data, test):
    loss_record = []
    min_loss = 100000

    epochs = 1000
    batch_size = 2000
    learning_rate = 0.01
    optimizer = optim.Adam(model_stable_no_proj.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_data), torch.tensor(test))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        loss_parts = []
        model_stable_no_proj.train()
        for data in train_dataloader:
            optimizer.zero_grad()
            loss, _ = runbatch(model_stable_no_proj, loss_stable, data, True)
            loss_parts.append(np.array([l.cpu().item() for l in loss]))

            optim_loss = loss[0] if isinstance(loss, (tuple, list)) else loss
            optim_loss.backward()
            optimizer.step()
            epoch_loss = sum(loss_parts) / len(dataset)
        print(f'Epoch {epoch}, loss: {epoch_loss}')
        loss_record.append(epoch_loss)
        if np.sum(epoch_loss)< min_loss:
            min_loss = np.sum(epoch_loss)
            torch.save(model_stable_no_proj.state_dict(), 'checkpoint/model_stable_no_proj.pth')
    return model_stable_no_proj

def train_simple(model_simple, X_data, test, loss_simple):
    epochs = 1000
    batch_size = 2000
    learning_rate = 0.01
    min_loss = 100000
    
    optimizer = optim.Adam(model_simple.parameters(), lr=learning_rate)
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_data), torch.tensor(test))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_record_simple = []
    for epoch in range(1, epochs + 1):
        loss_parts = []
        model_simple.train()
        for data in train_dataloader:
            optimizer.zero_grad()
            loss, _ = runbatch(model_simple, loss_simple, data)
            loss_parts.append(np.array([l.cpu().item() for l in [loss]]))

            optim_loss = loss[0] if isinstance(loss, (tuple, list)) else loss
            optim_loss.backward()
            optimizer.step()
            epoch_loss = sum(loss_parts) / len(dataset)
        print(f'Epoch {epoch}, loss: {epoch_loss}')
        loss_record_simple.append(epoch_loss)
        if epoch_loss< min_loss:
            min_loss = epoch_loss
            torch.save(model_simple.state_dict(), 'checkpoint/simple_nn.pth')
    return model_simple

if __name__ == "__main__":
    main()