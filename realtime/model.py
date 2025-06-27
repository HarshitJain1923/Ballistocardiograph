import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from collections import deque 
import matplotlib.pyplot as plt
import time
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class LSTM(torch.nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=256, batch_first=True)
        self.linear1 = torch.nn.Linear(256, 64)
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):  # x: (B, 6780, 1)
        h_t, _ = self.lstm(x)
        h_t = h_t[:, -1, :]  # Last time step
        x = self.linear1(h_t)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x.view(-1)  # (B,)
class LSTM2(nn.Module):
  def __init__(self):
    super(LSTM2, self).__init__()
    self.lstm = nn.LSTM(input_size = 6780, hidden_size = 512, batch_first = True)
    self.linear1 = nn.Linear(512, 64)
    self.dropout = nn.Dropout(0.5)
    self.linear2 = nn.Linear(64, 1)
    self.relu = nn.ReLU()

  def forward(self, x):
    h_t, c_t = self.lstm(x)
    h_t = h_t.squeeze()
    res = self.linear1(h_t)
    res = self.relu(res)
    res = self.dropout(res)
    res = self.linear2(res)
    res = res.T
    return res