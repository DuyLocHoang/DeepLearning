from lib import *
class L2Norm(nn.Module) :
    def __init__(self,input_channels = 512,scale = 20) :
        super(L2Norm,self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10
    
    def reset_parameters(self) :
        # Gan 20 vao trong tat ca cac weight -> Khoi tao
        nn.init.constant_(self.weight,self.scale)
    
    def forward(self, x) :
        #L2Norm
        # #x.size() = (batch_size,channels,height,width) 
        norm  = x.pow(2).sum(dim = 1, keepdim =true).sqrt() + self.eps # sum channels
        x = torch.divide(x,norm)
        #weight.size() = (512) -> (1,512,1,1) 
        weights = self.weight.unsquueze(0).unsquueze(2).unsquueze(3).expand_as(x)

        return weights*x

