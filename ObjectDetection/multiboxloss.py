# Jascard
# Hard negative mining: negative default box = 3 positive default box 
# Loss in regression : SmoothL1 Loss
# Loss in classification(multi class): Cross entropy

from lib import *
from utils.box_utils import *

class MultiBoxLoss(nn.Module) :
    def __init__(self,jaccard_threshold = 0.5, neg_pos = 3, device = "cpu") :
        super(MultiBoxLoss,self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.neg_pos = neg_pos
        self.device = device
    def foward(self,predictions,target) :
        loc_data, conf_data, dbox_list = predictions

        # loc_data(batch_num,num_dbox,num_class)
        num_batch  = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_class = loc_data.size(2)

        # De dung SmothL1 phai chuyen ve dang long
        conf_t_label = torch.longTensor(num_batch, num_dbox).to(device)
        loc_t = torch.Tensor(num_batch,num_dbox,4)

        for idx in range(num_batch):
            truths = target[idx][:,:,:-1].to(self.device) #(xmin,ymin,xmax,ymax)
            labels = target[idx][:,:,-1].to(self.device) #label

            dbox = dbox_list.to(self.device)
            variances = [0.1,0.2]
            match(self.jaccard_threshold,truths,dbox,variances,loc_t,conf_t_label,idx)