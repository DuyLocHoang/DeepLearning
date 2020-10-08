import numpy as np
#https://victorzhou.com/blog/intro-to-cnns-part-2/
class Conv3x3 :
    def __init__(self,num_filters) :
        self.num_filters = num_filters
    #divide 9 to reduce the variance
        self.filters = np.random.randn(num_filters,3,3)/9
    def iterate_regions(self,image) :
        h,w = image.shape
        for i in range(h-2) :
            for j in range(w-2) :
                im_region = image[i:(i+3),j:(j+3)]
                yield im_region,i,j
    def forward(self,input) :
        self.last_input = input
        h,w = input.shape
        output = np.zeros((h-2,w-2,self.num_filters))
        for im_region, i, j in self.iterate_regions(input) :
            #Do la ma tran anh co 32x32x3 tinh tren ca 3 kenh
            #np.sum([[[0,1,1],[0,5,5],[0,6,6]],[[0,1,1],[0,5,5],[0,6,6]],[[0,1,1],[0,5,5],[0,6,6]]],axis=(1,2)) solution : (24,24,24)
            output[i,j] = np.sum(im_region*self.filters, axis = (1,2))
        return output
    def backprop(self,d_L_d_out,learning_rate) :
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region,i,j in self.iterate_regions(self.last_input) :
            for f in range(self.num_filters) :
                d_L_d_filters[f] += d_L_d_out[i,j,f]*im_region
        self.filters -= learning_rate*d_L_d_filters
        return None  
