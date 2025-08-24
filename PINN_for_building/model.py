# This code works with BEAR dataset presented in paper 'BEAR-Data: Analysis and Applications of an Open Multizone Building Dataset'.
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

# For this example, we use cpu for simplicity.
use_cuda = False

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
BoolTensor = torch.cuda.BoolTensor if use_cuda else torch.BoolTensor
Tensor = FloatTensor
    
"""
System Idenfication: learning odefunc theta
Input:
    x0: initial state
    y: action, disturbance
    t: time index
Output:
    x: state trajectory
"""
class PhysicODE(nn.Module):
    def __init__(self, n=9):
        super(PhysicODE, self).__init__()
        self.rij = [nn.Parameter(torch.ones(1)*1e-4) for _ in range(n)]
        self.p = nn.Parameter(torch.randn(1))
        self.cp = nn.Parameter(torch.randn(1))
        self.tp =  nn.Parameter(Tensor([1.8]))
        self.encode = nn.Sequential(
            nn.Linear(2, 6), nn.Tanh(),
            nn.Linear(6, 1)
        )
        self.encode2 = nn.Sequential(
            nn.Linear(11, 12), nn.Tanh(),
            nn.Linear(12, 1)
        )
        self.step = torch.tensor(1.0)
        self.grid_constructor = self._grid_constructor_from_step_size(self.step)

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        """
        Inputs: step size
        Returns: _grid_constructor #to build a grid search table
        """
        def _grid_constructor(t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    def _step_func(self, y0, name=None):
        """
        Inputs:
            y0: (B, 11) -- current room + 9 room + action
        Returns:
            dx: (B, 1)
        """
        if name == 'C1': # Combined approach
            dx = torch.sum(torch.stack([1 * self.rij[i] * (y0[:,i+1:i+2]-y0[:,0:1]) for i in range(9)], dim=0), dim=0)
            dx += self.encode(torch.cat((y0[:,-1:], y0[:,0:1]),dim=1))
        elif name == 'P1': # Physic-Based approach
            dx = torch.sum(torch.stack([1 * self.rij[i] * (y0[:,i+1:i+2]-y0[:,0:1]) for i in range(9)], dim=0), dim=0)
            dx += self.cp * y0[:,-1:] * (torch.ones(1)*self.tp - y0[:,0:1]) + self.p
        elif name == 'D1': # Model-free approach
            dx = self.encode2(y0)
        return dx

    def forward(self, x0, y, t, name=None):
        """
        Integrate function
        Inputs:
            x0: (B, 1)       # all room's temperature + amb_temperature
            y:  (L, B, 1)    # current room action
            t:  (L)          # time (length)
        Returns:
            x: (L, B, 1)     # room's future temperature
        """
        time_grid = self.grid_constructor(t)
        solution = [x0]

        j = 1
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            y0 = torch.cat((x0, y[j-1]), dim=1)
            dx = self._step_func(y0, name=name)
            x1 = x0 + dx

            while j < len(t) and t1 >= t[j]:
                solution.append(self._linear_interp(t0, t1, x0, x1, t[j]))
                j += 1
            x0 = x1

        return torch.stack(solution)

    def _linear_interp(self, t0, t1, y0, y1, t):
        # Linear interpolation
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
    
def transform(x, length=2, day=14):
    """
    Transform the input data for unit conversion and reshaping.

    Args:
        x (numpy.ndarray): Input array of shape (T, day*B).
        length (int): Length of each segment.
        day (int): Number of days.

    Returns:
        numpy.ndarray: Transformed data reshaped and scaled.
    """
    # Transpose x to bring day*B to the first dimension for easier processing
    x = np.transpose(x, (1, 0, 2))  # Shape becomes (day*B, T, feature)

    # Initialize the transformed data array
    transformed_data = []

    # Loop through each day to process the data
    for d in range(day):
        # Collect segments for the current day based on length
        segments = x[[d + day * i for i in range(96 // length)]]
        # Reshape and flatten segments for the current day
        transformed_day = segments.reshape(-1)
        transformed_data.append(transformed_day)

    # Combine all days, scale, and offset
    result = np.concatenate(transformed_data) * 10 + 40
    return result

class BuildingDataset(Dataset):
    """
    A PyTorch dataset for building control tasks.
    """
    def __init__(self, file, room=0, N_train=7, length=10, resolution=3, normalize=True, train=True):
        """
        Initializes the dataset.

        Parameters:
        - file (str): Path to the dataset file (.npz format).
        - room (int): Index of the room to control (default is 0). Different rooms have different patterns, and different method may excel.
        - N_train (int): Number of days for training set. (default is 7).
        - length (int): Length of each sequence in time steps (default is 10).
        - resolution (int): Sampling resolution (e.g., every 3rd time step) (default is 3).
        - normalize (bool): Whether to normalize the data (default is True).
        - train (bool): Whether to load training or test data (default is True).
        """
        self.file = file
        data = np.load(file)
        if train:
            # Select training trajectories with specified resolution.
            xx = Tensor(np.transpose(data['state'], (1,0,2)))[::resolution, :N_train, :]   # shape (T, N, 10)
            uu = Tensor(np.transpose(data['action'],(1,0,2)))[::resolution, :N_train, room:room+1]  # shape (T, N, 1)
        else:
            xx = Tensor(np.transpose(data['state'], (1,0,2)))[::resolution, N_train:2*N_train, :]   # shape (T, N, 10)
            uu = Tensor(np.transpose(data['action'],(1,0,2)))[::resolution, N_train:2*N_train, room:room+1]  # shape (T, N, 1)
        if normalize:
            xx = (xx - 40) / 10
            uu = uu / 100
        # Prepare the input features `yy` by concatenating relevant states and actions.
        yy = torch.cat((xx[:,:,list(set(range(10))-set([room]))], uu), dim=-1) # (T, N, 10)
        T = xx.shape[0]

        # Initialize lists to store processed data.
        x0 = []  # Initial state for each sequence.
        truex = []  # True states for the specified room.
        y = []  # Features for training the model.

        # Slice the data into sequences of the specified length.
        for t in range(0, T - length + 1, length):
            x0.append(xx[t, :, room:room+1])  # Initial state for this sequence.
            truex.append(xx[t:t+length, :, room:room+1])  # True trajectory for this sequence.
            y.append(yy[t:t+length-1])  # Features for predicting the next state.
    
        self.x0 = torch.cat(x0, dim=0)                     # (B, 1)
        self.truex = torch.cat(truex, dim=1)            # (L+1, B, 1)
        self.y = torch.cat(y, dim=1)                    # (L+1, B, 1)
        print('Prepare dataset...: x.shape', self.truex.shape)

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        x0 = self.x0[idx]
        y = self.y[:,idx]
        truex = self.truex[:, idx]
        return x0, y, truex