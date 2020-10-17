from lib import *
from make_datapath import *
from utils.augmentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort,\
     Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans
from extract_inform_annotation import *
# Compose : 
# ConvertFromInts:
# ToAbsoluteCoords: De lay cac thong so ve pixel thuc su tren buc anh
# PhotometricDistort : Doi cac thong so ve mau sac
# Expand : khi mo rong anh thi chen cac thong so ve khoang den
# ToPercentCoords : chuyen pixel that su chuyen ve (0,1)
# SubtractMeans : tru cac gia tri trung binh cua moi channel

class DataTransform() :
    def __init__(self,input_size,color_mean):
        self.data_transform = {
            'train' : Compose([
                # Muon xu ly anh phai chuyen ve size that cua buc anh
                # sau khi xu ly xong chuyen ve dang quy chuan (0,1)
                ConvertFromInts(), # Convert img from int to float32
                ToAbsoluteCoords(), # back annotation to normal type
                PhotometricDistort(), # Change color using random
                Expand(color_mean), # Keo dai buc anh 
                RandomSampleCrop(), #random crop image
                RandomMirror(), # Xoay anh 180 do nhu nhin trong guong
                ToPercentCoords(), # Chuan hoa annotation ve dang [0,1]
                Resize(input_size),
                SubtractMeans(color_mean) # subtract mean cua color BGR 
            ]),
            'val' : Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
        }
    def __call__(self,img,phase,boxes,label) :
        return self.data_transform[phase](img,boxes,label)

if __name__ == "__main__" :
    # prepare train, val, annotation list
    root_path = "./data/VOC2012/"
    train_img_list,train_annotation_list,val_img_list,val_annotation_list = make_datapath_list(root_path)
    # read img
    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path) # return picture BGR (height,width,channels)
    height,width,channels = img.shape

    # annotation information
    classes = ["bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa",
    "train","tvmonitor"]
    train_anno = Anno_xml(classes)
    anno_infor_list = train_anno(train_annotation_list[0],width,height)

    # plot original image 
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # mac dinh cua matplotlib la RGB
    plt.show()
    
    #prepare data transform
    color_mean = (104,117,123)
    input_size = 300
    transform = DataTransform(input_size,color_mean)

    # transform train img
    phase = "train"
    img_transform, boxes, labels = transform(img,phase,anno_infor_list[:,:4],anno_infor_list[:,4])
    plt.imshow(cv2.cvtColor(img_transform,cv2.COLOR_BGR2RGB)) # mac dinh cua matplotlib la RGB
    plt.show()
    # transform val img
    phase = "val"
    img_transform, boxes, labels = transform(img,phase,anno_infor_list[:,:4],anno_infor_list[:,4])
    plt.imshow(cv2.cvtColor(img_transform,cv2.COLOR_BGR2RGB)) # mac dinh cua matplotlib la RGB
    plt.show()