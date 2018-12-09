import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.cuda.nccl as nccl
import numpy
import scipy.io as sio

torch.manual_seed(1)   # reproducible

# Hyper Parameters
EPOCH = 10              # number of epochs
BATCH_SIZE = 4         # used for training 
BATCH_SIZE2 = 256      # used for test
LR = 1e-5              # learning rate
MM = 0                 # momentum - used only with SGD optimizer

# use 3 dropout values: one for the input image, one for convolution layers, one for final (linear) layers
DROPOUT_INITIAL = 0
DROPOUT_MIDDLE = 0         
DROPOUT_CLASSIFIER = 0.9

RELU = True

L_FIRST = 103
L_SECOND = 128
L_THIRD = 192
L_FOURTH = 368
L_FIFTH = 192
L_SIXTH = 96
L_SEVENTH = 32

TRAIN_VECTOR_SIZE = 200
TRAIN_SIZE = 1800        # 3437
TEST_SIZE = 40976

test = sio.loadmat('database200.mat')

test_data = test['test_data']
test_data = numpy.array(test_data, dtype=numpy.float32)
test_data = torch.from_numpy(numpy.array(test_data))
test_data = test_data.permute(0,3,1,2)

test_target = test['test_target']
test_target = numpy.array(test_target, dtype=numpy.float32)
test_target = torch.from_numpy(numpy.array(test_target))

train_data = test['train_data']
train_data = numpy.array(train_data, dtype=numpy.float32)
train_data = torch.from_numpy(numpy.array(train_data))
train_data.requires_grad = True
train_data = train_data.permute(0,3,1,2)

train_target = test['train_target']
train_target = numpy.array(train_target, dtype=numpy.float32)
train_target = torch.from_numpy(numpy.array(train_target))


# Data Loader for easy mini-batch return in training
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2 )
test_loader  = Data.DataLoader(dataset =  test_data, batch_size = BATCH_SIZE2, shuffle = False, num_workers = 2 )

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(         
            nn.Dropout(p = DROPOUT_INITIAL),
            
            nn.Conv2d(L_FIRST, L_SECOND, 5, 1, 3),
            nn.Dropout(p = DROPOUT_MIDDLE),
            nn.ReLU(RELU),                      
            nn.MaxPool2d(2),    
            nn.BatchNorm2d(L_SECOND),

            nn.Conv2d(L_SECOND, L_THIRD, 5, 1, 3),     
            nn.Dropout(p = DROPOUT_MIDDLE),             
            nn.ReLU(RELU),                      
            nn.MaxPool2d(2),                
            nn.BatchNorm2d(L_THIRD),

#            nn.Conv2d(L_THIRD, L_FOURTH, 5, 1, 3),     
#            nn.Dropout(p = DROPOUT_MIDDLE),             
#            nn.ReLU(RELU),                      
#            nn.MaxPool2d(2),                 
#            nn.BatchNorm2d(L_FOURTH),

#            nn.Conv2d(L_FOURTH, L_FIFTH, 5, 1, 3),     
#            nn.Dropout(p = DROPOUT_MIDDLE),             
#            nn.ReLU(RELU),                      
#            nn.MaxPool2d(2),                 
#            nn.BatchNorm2d(L_FIFTH),
        )
        
        self.conv2 = nn.Sequential(         
            nn.Dropout(p = DROPOUT_INITIAL),
            
            nn.Conv2d(L_FIRST, L_SECOND, 5, 1, 3),
            nn.Dropout(p = DROPOUT_MIDDLE),
            nn.ReLU(RELU),                      
            nn.MaxPool2d(2),    
            nn.BatchNorm2d(L_SECOND),

            nn.Conv2d(L_SECOND, L_THIRD, 5, 1, 3),     
            nn.Dropout(p = DROPOUT_MIDDLE),             
            nn.ReLU(RELU),                      
            nn.MaxPool2d(2),                
            nn.BatchNorm2d(L_THIRD),
        )       
        
        
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT_CLASSIFIER),
            nn.Linear(L_FIFTH*4,L_SIXTH),
            nn.ReLU(RELU),
            
            nn.Dropout(DROPOUT_CLASSIFIER),
            nn.Linear(L_SIXTH, L_SEVENTH),
            nn.ReLU(RELU),
            
            nn.Linear(L_SEVENTH, 10),
        )

    def forward(self, x):
        x1 = self.conv(x)          
        x2 = self.conv2(x)  
        x = x1 + x2
        x = x.view(x.size(0), -1) 
        output1 = self.classifier(x)
        return output1, x 
    

cnn = CNN()
print(cnn)  # net architecture

#torch.cuda.synchronize()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  cnn = nn.DataParallel(cnn)

cnn = torch.nn.DataParallel(cnn, device_ids=[0]).cuda()

cnn.to(device)

import torch.optim as optim

# it seems these loss functions have a different form of return 
#loss_func = nn.L1Loss()
#loss_func = nn.MSELoss()
#loss_func = nn.SoftMarginLoss()
#loss_func = nn.MultiLabelSoftMarginLoss()
#loss_func = nn.PoissonNLLLoss()


loss_func = nn.CrossEntropyLoss()
#loss_func = nn.NLLLoss()


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
#optimizer = optim.SGD(cnn.parameters(), lr=LR, momentum=MM)

train_target = torch.t(train_target).type(torch.LongTensor).cuda()
test_target = torch.t(test_target).type(torch.LongTensor).cuda()

# training 
for epoch in range(EPOCH):
    loss = None
    for step, data in enumerate(train_loader,0):
        
        train_data, train_target = train_data.to(device), train_target.to(device)
        
        optimizer.zero_grad()           # clear gradients for this training step            
        output = cnn(train_data)[0]        # cnn output
        loss = loss_func(output, train_target[0])   # cross entropy loss
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(train_data)
            pred_y = torch.max(test_output, 1)[1]
            accuracy = torch.sum(pred_y == train_target[0]).type(torch.FloatTensor) / float(train_target.size(1))
            print('Epoch: ', epoch, '| train loss: ' % loss.data.cpu().numpy(), '| training accuracy: %.3f' % accuracy)     
#label1 = torch.zeros(1,9)
# pixels in each class in the test vector
label1 = torch.tensor([6431, 18449, 1899, 2864, 1145, 4829, 1130, 3482, 747]).type(torch.FloatTensor)
label2 = torch.zeros(1,9)

test_data, test_target = test_data.to(device), test_target.to(device)

#with torch.no_grad():
outputs = cnn(test_data)[0]
_, predicted = torch.max(outputs, 1)
predicted = predicted.to(device)

c = (predicted == test_target[0]).squeeze()
c = c.to(device)
    
for j in range (40976):
    if (c[j] == 1):
        label2[0,predicted[j]-1] += 1                 
                
                               
#print('Correct classification in each class: ',label2)
percent = (torch.sum(c).item()/TEST_SIZE)
print('Correct Classification (Percent): %.2f' % percent)
print('Results by classes: ',label2/label1)

torch.cuda.empty_cache()
