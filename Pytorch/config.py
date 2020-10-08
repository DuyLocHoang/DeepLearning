from lib import *

# Khoi tao cai so 1234 de khi thuc hien tren cac may khac thi van cung ket qua nhu nhau neu nhu cung 1 so 1234
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


resize = 224 
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]

num_epoch = 2
batch_size = 4 
save_path = './weight_finetunning.pth'