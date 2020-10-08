from lib import *
from config import *
from utils import make_datapath_list, train_model, params_to_update, load_model
from ImageTranform import ImageTransform
from dataset import MyDataset

#Tranfers learning : Su dung cho dataset nho, chi thay doi layer cuoi cung cua network
#Finetunning : Su dung cho dataset nho , thay hoi hau nhu toan bo network.

def main() :
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    #dataset
    train_dataset = MyDataset(train_list,transform=ImageTransform(resize,mean,std),phase='train')
    val_dataset = MyDataset(val_list,transform=ImageTransform(resize,mean,std),phase='val')

    #dataloader
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size,shuffle=False)
    dataloader_dict = {"train":train_dataloader,"val":val_dataloader}

    #NETWORK
    use_pretrained = True
    net = models.vgg16(pretrained = use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2,bias=True)
    print(net)
    #setting mode
    net = net.train()

    #LOSS
    criterior = nn.CrossEntropyLoss()

    #OPTIMIZER
    # Update thong so mong muon
    # params_to_update = []
    # update_params_name = ["classifier.6.weight","classifier.6.bias"]
    # for name,param in net.named_parameters() :
    #     if name in update_params_name :
    #         param.requires_grad = True
    #         params_to_update.append(param)
    #         print(name)
    #     else:
    #         param.requires_grad = False
    params1,params2,params3 = params_to_update(net)
    # params = update trong so luu vao
    #lr = he so hoc
    #momentun = 
    optimizer = optim.SGD([
        {"params" :params1, "lr" : 1e-4},
        {"params" :params2, "lr" : 5e-4},
        {"params" :params3, "lr" : 1e-3} 
    ],momentum = 0.9)

    #training

    train_model(net,dataloader_dict,criterior, optimizer, num_epoch)

if __name__ == "__main__" :
    main()

    #network
    # use_pretrained = True
    # net = models.vgg16(pretrained = use_pretrained)
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=2,bias=True)
    # load_model(net,save_path)