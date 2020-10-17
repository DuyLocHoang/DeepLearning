from lib import *
from l2norm import *
from default_box import *
def create_vgg() :
    layers = []
    input_channels = 3
    cfgs = [64,64,'M',128,128,'M',
            256,256,256,'MC',512,512,512,'M',
            512,512,512]
    for cfg in cfgs :
        if cfg == 'M' : #Floor
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        elif cfg == 'MC' : # Celling (lam tron len)
            layers += [nn.MaxPool2d(kernel_size = 2 ,stride = 2, ceil_mode = True)]
        else :
            conv2d = nn.Conv2d(input_channels, cfg, kernel_size = 3, padding = 1)
            layers += [conv2d, nn.ReLU(inplace = True)] # inplace =True : Xac dinh viec khong luu dau vao ReLU vao ReLU
            input_channels = cfg
    pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
    conv6 = nn.Conv2d(512,1024,kernel_size = 3, padding = 6, dilation = 6)
    conv7 = nn.Conv2d(1024,1024, kernel_size = 1)
    layers += [pool5,conv6,nn.ReLU(inplace = True),conv6, nn.ReLU(inplace = True)]
    return nn.ModuleList(layers) 

def create_extras() :
    layers = []
    input_channels = 1024
    cfgs = [256,512,128,256,128,256,128,256]
    layers += [nn.Conv2d(input_channels,cfgs[0], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[0],cfgs[1], kernel_size = 3,stride =2, padding =1)]
    layers += [nn.Conv2d(cfgs[1],cfgs[2], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[2],cfgs[3], kernel_size = 3,stride =2, padding =1)] 
    layers += [nn.Conv2d(cfgs[3],cfgs[4], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[4],cfgs[5], kernel_size = 3)]
    layers += [nn.Conv2d(cfgs[5],cfgs[6], kernel_size = 1)]
    layers += [nn.Conv2d(cfgs[6],cfgs[7], kernel_size = 3)]
    return nn.ModuleList(layers)

def create_loc_conf(num_classes = 21,bbox_ratio_num = [4,6,6,6,4,4]) :
    loc_layers = []
    conf_layers = []
    #source 1
    loc_layers += [nn.Conv2d(512,bbox_ratio_num[0]*4,kernel_size = 3,padding = 1)]
    conf_layers += [nn.Conv2d(512,bbox_ratio_num[0]*num_classes,kernel_size = 3,padding = 1)]
    #source 2
    loc_layers += [nn.Conv2d(1024,bbox_ratio_num[1]*4,kernel_size = 3,padding = 1)]
    conf_layers += [nn.Conv2d(1024,bbox_ratio_num[1]*num_classes,kernel_size = 3,padding = 1)]
    #source 3
    loc_layers += [nn.Conv2d(512,bbox_ratio_num[2]*4,kernel_size = 3,padding = 1)]
    conf_layers += [nn.Conv2d(512,bbox_ratio_num[2]*num_classes,kernel_size = 3,padding = 1)]
    #source 4
    loc_layers += [nn.Conv2d(256,bbox_ratio_num[3]*4,kernel_size = 3,padding = 1)]
    conf_layers += [nn.Conv2d(256,bbox_ratio_num[3]*num_classes,kernel_size = 3,padding = 1)]
    #source 5
    loc_layers += [nn.Conv2d(256,bbox_ratio_num[4]*4,kernel_size = 3,padding = 1)]
    conf_layers += [nn.Conv2d(256,bbox_ratio_num[4]*num_classes,kernel_size = 3,padding = 1)]
    #source 6
    loc_layers += [nn.Conv2d(256,bbox_ratio_num[5]*4,kernel_size = 3,padding = 1)]
    conf_layers += [nn.Conv2d(256,bbox_ratio_num[5]*num_classes,kernel_size = 3,padding = 1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)

cfg = {
    'num_classes' :21,
    'input_size' : 300,
    "bbox_aspect_num" : [4,6,6,6,4,4],
    'feature_maps' :[38,19,10,5,3,1],
    'steps' : [8,16,32,64,100,300], # Do lon cua cac default box
    'min_size' : [60,111,162,213,240,264], #Size of default box
    'max_size' : [111,162,213,240,264,270],
    'aspect_ratio' : [[2],[2,3],[2,3],[2,3],[2],[2]]
    }

class SSD(nn.Module) :
    def __init__(self,phase,cfg) :
        super(SSD,self).__init__()
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        #create main module
        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc,self.conf = create_loc_conf(self.num_classes,cfg['bbox_aspect_num'])
        self.l2norm = L2Norm()
        # create default box
        dbox = DefBox(cfg)
        self.dbox_list = dbox.create_defbox()
        
        if phase == "inference" :
            self.detect = Detect()
    def forward(self,x) :
        source = list()
        loc = list()
        conf = list()

        for k in range(23) :
            x = self.vgg[k][x]
        #source 1
        source1 = self.l2norm(x)
        source.append(source1)

        for k in range(23,len(self.vgg)) :
            x = self.vgg[k][x]       
        # source 2
        source.append(x)
        # source 3 -> source 6
        for k, v in enumerate(self.extras) :
            x = nn.ReLU(v(x),inplace = True)
            if k%2 != 0 :
                source.append(x)
        for (x,l,c) in zip(source, self.loc,self.conf) :
            #(batch_size,4*aspect_ratio_num,featuremap_height,feature_map_width)
            #=> (batch_size,featuremap_height,feature_map_width,,4*aspect_ratio_num)
            loc.append(l(x).permute(0,2,3,1).contiguous()) # chuyen doi dimension trong permute ko duoc sap xep lien tuc tren memory nen can ham contiguous()
            conf.append(c(x).permute(0,2,3,1).contiguous())
        
        #(batch_size,4*8732)
        loc = torch.cat([o.view(o.size(0),-1) for o in loc],1)
        #(batch_size,21*8732)
        conf = torch.cat([o.view(o.size(0),-1) for o in conf],1)
        loc = loc.view(loc.size(0),-1,4) #(batch_size,8732,4)
        conf = conf.view(conf.size(0),-1,self.num_classes) #(batch_size,8732,21)
        
        output = (loc,conf,self.dbox_list)
        if phase == "inference" :
            return self.detect(output[0],output[1],output[2])
        else :
            return output

def decode(loc,defbox_list) :
    
    """
    parameters:
    loc : [8732,4]  (delta_x,delta_y,delta_w,delta_h)
    defbox_list: [8732,4]   (cx_d,cy_d,w_d,h_d)
    output :
    boxes : [xmin,ymin,xmax,ymax]
    """
    boxes = torch.cat((
        defbox_list[:,:2]+0.1*loc[:,:2]*defbox_list[:,2:],
        defbox_list[:,2:]*torch.exp(loc[:,2:]*0.2)),dim = 1)
    
    boxes[:,:2] -= boxes[:,2:]/2 # calculate xmin,ymin
    boxes[:,2:] += boxes[:,:2]   # calculate xmax,ymax
    
    return boxes

def nms(boxes, scores, overlap =0.45, top_k = 200 ) :
    """
    boxes : [num_box, 4]
    scores : [num_box]
    """
    count = 0
    keep = scores.new(scores.size(0)).zeros_().long() # giu id bounding box
    # box coordinate
    x1 = boxes[:,0] # xmin
    y1 = boxes[:,1] # ymin
    x2 = boxes[:,2] # xmax
    x2 = boxes[:,3] # ymax
    # Dien tich cua bbox
    area = torch.mul(x2-x1,y2-y1)
    
    tmp_x1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value,idx = scores.sort(0)
    idx = idx[-top_k:] # id cua top_k box co do tu tin cao nhat

    while idx.numel() > 0 :
        # id cua box co do tu tin cao nhat
        i = idx[-1]
        keep[count]
        count += 1
        if idx.size(0) == 1 :
            break
        idx = idx[:,-1] # id cua boxes ngoai tru box co do tu tin cao nhat

        #infor boxes
        # Lay tat ca cac index tu trong x1 va gan vao tmp_x1
        torch.index_select(x1,0,idx, out = tmp_x1)
        torch.index_select(y1,0,idx, out = tmp_y1)
        torch.index_select(x2,0,idx, out = tmp_x2)
        torch.index_select(y2,0,idx, out = tmp_y2)

        # Lay cac gia tri xmin,ymin,xmax,ymax cua phan overlap
        tmp_x1 = torch.clamp(tmp_x1, min = x1[i]) # x1[i] if  tmp_x1 < x[1]
        tmp_y1 = torch.clamp(tmp_y1, min = y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max = x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max = y2[i])
        
        # reisize tra ve luon cho tmp_w
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1 # Chieu rong
        tmp_h = tmp_y2 - tmp_y1 # Chieu cao

        # Dua nhung phan tu nho hon 0 ve 0.0
        tmp_w = torch.clamp(tmp_w,min = 0.0)
        tmp_h = torch.clamp(tmp_h,min = 0.0)

        #area overlap
        inter = tmp_w*tmp_h

        # lay ra area cua cac phan tu idx
        others_area = torch.index_select(area,0,idx)
        union = area[i] +others_area - inter
        iou = inter / union

        #Nho hon overlap= 0.45 thi giu lai 
        idx = idx[iou.le(overlap)]
    
    # count : so luong cac box giu lai dc 
    return keep,count

class Detect(Function) :
    def __init__(self,conf_thresh = 0.01, top_k =200, nms_thresh = 0.45) :
        super(Detect,self).__init__()
        self.softmax = nn.Softmax(dim = -1) # Lay ra softmax o dimesion cuoi cung
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
    def forward(self,loc_data,conf_data,dbox_list) :
        num_batch = loc_data.size(0) # batch size
        num_box = loc_data.size(1)  #8732
        num_classes = conf_data.size(2) #21 
        
        conf_data = self.softmax(conf_data) # bactch size, number box, number_class -> batch_num,num_class, number_box
        conf_preds = conf_data.transpose(0,2,1)

        output = torch.zeros_(num_batch,num_classes,self.top_k,5)
        # xu ly tung buc anh trong 1 batch
        for i in range(num_batch) :
            # Tinh bbox tu offset information va defaul box
            decode_boxes = decode(loc_data[i],dbox_list)
            # copy conference score cua anh thu i
            conf_scores = conf_preds[i].clone()

            for cl in range(1,num_classes) :
                c_mask = conf_preds[cl].gt(self.conf_thresh) # Lay ra nhung confidence lon hon conf_thresh
                scores = conf_preds[cl][c_mask]
                if scores.numel() == 0: #nelement()
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes) #(8732,4)
                boxes = decode_boxes[l_mask].view(-1,4)
                ids, count = nms(boxes,scores,self.nms_thresh,self.top_k)

                output[i,cl,:count] =torch.cat((scores[ids[:count]].unsqueeze(1),boxes[ids[:count]]),1)
        return output

if __name__ == "__main__" :
    # vgg = create_vgg()
    # extras = create_extras()
    # loc,conf = create_loc_conf()
    #  print(vgg)
    # print(extras)
    # print(loc)
    # print(conf)
    ssd = SSD("train", cfg)
    print(ssd)