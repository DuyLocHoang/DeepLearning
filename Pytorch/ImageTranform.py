from lib import *
class ImageTransform() :
    def __init__(self,resize,mean,std) :
        self.data_transform = {
            'train': transforms.Compose([
                # random moi mot lan dua buc anh vao thi se ra 1 kq khac dan den tinh da dang cho data
                # lam cho model hoc duoc nhieu hon
                # chinh xac hon cho cac truong hop trong thuc te
                transforms.RandomResizedCrop(resize,scale = (0.5,1.0)),
                # Xac suat de xoay 1 buc anh 
                transforms.RandomHorizontalFlip(),
                # Phai de ve dang tensor thi network moi training
                transforms.ToTensor(),
                # Chuan hoa de chinh xac hon
                transforms.Normalize(mean,std)
            ]),
            'val': transforms.Compose([
                #ko su dung random vi khi luc test kq validation se khong con chinh xac nua 
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
                
            ]),
            'test' : transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])

        }
    def __call__(self,img, phase = 'train') :
        #self.data_transform[phase] : Goi ra train or val
        return self.data_transform[phase](img)
