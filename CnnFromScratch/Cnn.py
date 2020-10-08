import mnist
import numpy as np
from Conv import Conv3x3
from softmax import Softmax
from PoolingLayer import MaxPool2

train_images = mnist.train_images()[:1000]
train_labels = mnist.train_labels()[:1000]
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]
conv = Conv3x3(8)
pool = MaxPool2()
softmax = Softmax(13*13*8,10)
def forward(image,labels) :
    output = conv.forward((image/255)-0.5)
    output = pool.forward(output)
    output = softmax.forward(output)

    loss = -np.log(output[labels])
    acc = 0
    acc = 1 if np.argmax(output) == labels else 0
    return output, loss ,acc

def train(im,labels,learning_rate = 0.005) :
    out, loss, acc = forward(im,labels)
    gradient = np.zeros(10)
    gradient[labels] = -1/out[labels]

    gradient = softmax.backprop(gradient,learning_rate)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient,learning_rate)
    return loss,acc

# Train and Test
for epoch in range(3) :
    print('--- Epoch %d ---' % (epoch + 1))
    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    #Train
    loss = 0
    num_correct = 0
    for i, (im,labels) in enumerate(zip(train_images,train_labels)) :

        if i> 0 and i%100 == 99 :
            print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))
            loss = 0
            num_correct = 0

        l ,acc = train(im,labels)
        loss += l
        num_correct +=acc

#Test 
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im,labels in zip(test_images,test_labels) :
    _, l,acc = forward(im,labels)
    loss += l
    num_correct += acc
num_tests = len(test_images)
print("Test Loss: " ,loss/num_tests)
print("Test Accuracy: ",num_correct/num_tests)
