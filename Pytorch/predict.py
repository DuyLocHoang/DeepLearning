from lib import *
from config import *
from utils import *
from ImageTranform import *

class_index = ['ants','bees']

class Predictor() :
    def __init__(self,class_index) :
        self.class_index = class_index
    def predict_max(self, out) :
        maxid = np.argmax(out.detach().numpy())
        predict_label_name = self.class_index[maxid]
        return predict_label_name

predictor = Predictor(class_index)
def predict(img) :
    # Call model 
    use_pretrained = True
    net = models.vgg16(pretrained = use_pretrained)
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)
    net.eval()
    # Load model 
    model = load_model(net,save_path)
    # prepare input image
    transform = ImageTransform(resize,mean,std)
    img = transform(img,phase = 'test')
    #(channels,height,weight) -> (batch size = 1, channel, height, weight)
    img = img.unsqueeze_(0)

    # predict 
    
    output = model(img)
    response = predictor.predict_max(output)
    return response

