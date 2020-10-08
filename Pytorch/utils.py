from lib import *
from config import *

def make_datapath_list(phase = "train") :
    rootpath = './data/hymenoptera_data/'
    target_path = osp.join(rootpath+phase+"/**/*.jpg")
#     print(target_path)

    path_list = []
    # List tat ca cac duong link co dinh dang nhu target_path
    for path in glob.glob(target_path) :
        path_list.append(path)
    return path_list  

def train_model(net,dataloader_dict,criterior,optimizer,num_epochs) :
    #Phan biet may co GPU hay ko
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ",device)
    for epoch in range(num_epochs) :
        print ("Epoch{}/{}".format(epoch,num_epochs))

        # Neu co GPU se chuyen vao trong GPU chay
        net.to(device)
        # Tang toc do tinh toan cua GPU
        torch.backends.cudnn.benchmark = True

        for phase in ["train","val"] :
            if phase == "train" :
                net.train()
            else :
                net.eval()
            epoch_loss = 0.0
            epoch_acc = 0
            if (epoch == 0) and (phase == "train") :
                continue
            #tqdm() : hien ra load toi dau
            for inputs,labels in tqdm(dataloader_dict[phase]) :
                
                # Move inputs, labels to GPU or CPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train") :
                    outputs = net(inputs)
                    loss = criterior(outputs,labels)
                    #Tim gia tri lon nhat trong moi hang
                    _,preds =torch.max(outputs,1)
                    
                    if phase == "train" :
                        loss.backward()
                        #Update parameter cho thang optimizer
                        optimizer.step()
                    #Tim tb loss cua cac anh vd 4 anh
                    epoch_loss += loss.item()*inputs.size(0)
                    # Tuy nhien labels tra ve tensor 
                    # Do tensor nen lay ra phai .data
                    epoch_acc += torch.sum(preds == labels.data) 
            epoch_loss = epoch_loss/len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_acc.double()/ len(dataloader_dict[phase].dataset)
            print('{} Loss: {} \n Acc: {}'.format(phase,epoch_loss,epoch_acc))
    torch.save(net.state_dict(),save_path)

def params_to_update(net) :
    params_to_update1 = []
    params_to_update2 = []
    params_to_update3 = []
    update_param_name_1 = ["feature"]
    update_param_name_2 = ["classifier.0.weight","classifier.0.bias","classifier.3.weight","classifier.3.bias"]
    update_param_name_3 = ["classifier.6.weight","classifier.6.bias"]
    for name, param in net.named_parameters() :
        if name in update_param_name_1  :
            param.requires_grad = True
            params_to_update1.append(param)
        elif name in update_param_name_1 :
            params.requires_grad = True
            params_to_update2.append(param)
        elif name in update_param_name_3 :
            param.requires_grad = True
            params_to_update3.append(param)
        else:
            param.requires_grad = False
    return params_to_update1,params_to_update2,params_to_update3

def load_model(net,model_path) :
    # chua fix parameter vao trong network nen can fix vao net work
    load_weight = torch.load(model_path)
    net.load_state_dict(load_weight)
    print(net)
    return net
    for name, param in net.named_parameters() :
        print(name,param)
    # Neu can chay tren GPU
    # load_weight = torch.load(model_path, map_location = ('cuda :0','cpu'))
    # net.load_state_dict(load_weight)
