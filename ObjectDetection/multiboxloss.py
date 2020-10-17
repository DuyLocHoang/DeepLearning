# Jascard
# Hard negative mining: negative default box = 3 positive default box 
# Loss in regression : SmoothL1 Loss
# Loss in classification(multi class): Cross entropy

from lib import *

class MultiBoxLoss(nn.Module) :
    def __init__(self,jaccard_threshold = 0.5, neg_pos = 3, device = "cpu") :
        super(MultiBoxLoss,self).__init__()
        self.jaccard_threshold = jaccard_threshold
        self.neg_pos = neg_pos
        self.device = device
    def foward(self,predictions,target) :
        