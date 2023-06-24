import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
  def __init__(self,input_size):
    self.input_size =input_size
    self.layer1 = nn.LSTM()