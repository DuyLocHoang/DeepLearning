#Liet ke cac path chua buc anh
import glob
import os.path as osp
# Xoay anh voi ty le ngau nhien
import random
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
# Chua cac thuat toan toi uu, update para
import torch.optim as optim 
# Chua cac ham so dieu khien data
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
from tqdm import tqdm