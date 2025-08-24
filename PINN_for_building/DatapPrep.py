

import torch
import numpy as np

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

torch.manual_seed(1234)


