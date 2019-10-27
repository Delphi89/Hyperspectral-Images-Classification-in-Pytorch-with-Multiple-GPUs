import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self, patience=40, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            #print('Resetting Patience Counter')

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Average Validation Loss Decreased ({self.val_loss_min:.2f} --> {val_loss:.2f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
        
        
class ReducedRate:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self, waiting=10, verbose=False):
        self.waiting = waiting
        self.verbose = verbose
        self.counter2 = 0
        self.best_score2 = None
        self.early_stop2 = False
        self.val_loss_min2 = np.Inf

    def __call__(self, val_loss, model):

        score2 = -val_loss

        if self.best_score2 is None:
            self.best_score2 = score2
        elif score2 < self.best_score2:
            self.counter2 += 1
            print(f'ReducedRate counter: {self.counter2} out of {self.waiting}')
            if self.counter2 >= self.waiting:
                self.early_stop2 = True
                self.counter2 = 0
            else:
                self.early_stop2 = False    
        else:
            self.best_score2 = score2
            self.counter2 = 0
            self.early_stop2 = False
            #print('Resetting Patience Counter')
            

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.cuda.nccl as nccl
import numpy
import numpy as np
import scipy.io as sio
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
#from pytorchtools import EarlyStopping

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
#print(torch.cuda.nccl.is_available())
torch.manual_seed(0)   # reproducible


# classification: % pixels /  average


# Counter for the execution time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

patience = 20              # if validation loss not going down, wait "patience" number of epochs
waiting = 10
train_losses = []         # to track the training loss as the model trains
#valid_losses = []         # to track the validation loss as the model trains
#avg_train_losses = []     # to track the average training loss per epoch as the model trains
#avg_valid_losses = []     # to track the average validation loss per epoch as the model trains
early_stopping = EarlyStopping(patience=patience, verbose=True)  # initialize the early_stopping object
reduced_rate = ReducedRate(waiting=waiting, verbose=True)

# Hyper Parameters
EPOCHS = 1              # number of epochs
BATCH_SIZE = 256         # used for training 
#BATCH_SIZE2 = 64         # used for test
LR = 1e-4                  # learning rate
MM = 0.99                  # momentum - used only with SGD optimizer

# use 3 dropout values: one for the input image, one for convolution layers, one for final (linear) layers
DROPOUT_INIT = 0.4
DROPOUT_MIDDLE_1 = DROPOUT_INIT
DROPOUT_MIDDLE_2 = DROPOUT_INIT
DROPOUT_CLASSIFIER = DROPOUT_INIT
DROPOUT_SKIP = DROPOUT_INIT
REGULARIZATION = LR

RELU = False

L_FIRST = 103
L_SECOND = 256
L_THIRD = L_SECOND
#L_FOURTH = 32
L_FIFTH = L_THIRD 
L_SIXTH = L_THIRD
L_SEVENTH = L_THIRD * 4  

TRAIN_VECTOR_SIZE = 200
TRAIN_SIZE = 9*TRAIN_VECTOR_SIZE     
VALIDATION_SIZE = 9*2*TRAIN_VECTOR_SIZE
TEST_SIZE = 42776-TRAIN_SIZE-VALIDATION_SIZE
min_val_loss = 999999

accuracy = 0
last_training_loss = 0

np.set_printoptions(threshold=np.inf)

# number of pixels for test in each class
label1 = torch.tensor([6031, 18049, 1499, 2464, 745, 4429, 730, 3102, 347]).type(torch.FloatTensor) 
label2 = torch.zeros(1,9)

# data is read from the file created in matlab
database = sio.loadmat('database200ivi.mat')

# data needed for the test is:
# - changed to numpy type
# - changed to torch type
# - the grad is removed 
# - the form of matrix is changed for [number_of_elements, channels, height, width] 
test_data = database['test_data']
test_data = numpy.array(test_data, dtype=numpy.float32)
test_data = torch.from_numpy(numpy.array(test_data))
test_data.requires_grad = False
test_data = test_data.permute(0,3,1,2)

test_target = database['test_target']
test_target = numpy.array(test_target, dtype=numpy.float32)
test_target = torch.from_numpy(numpy.array(test_target))
test_target.requires_grad = False
test_target = torch.t(test_target).type(torch.LongTensor).cuda()

train_data = database['train_data']
train_data = numpy.array(train_data, dtype=numpy.float32)
train_data = torch.from_numpy(numpy.array(train_data))
train_data.requires_grad = False
train_data = train_data.permute(0,3,1,2)

train_target = database['train_target']
train_target = numpy.array(train_target, dtype=numpy.float32)
train_target = torch.from_numpy(numpy.array(train_target))
train_target.requires_grad = False
train_target = torch.t(train_target).type(torch.LongTensor).cuda()

validation_data = database['validation_data']
validation_data = numpy.array(validation_data, dtype=numpy.float32)
validation_data = torch.from_numpy(numpy.array(validation_data))
validation_data.requires_grad = False
validation_data = validation_data.permute(0,3,1,2)

validation_target = database['validation_target']
validation_target = numpy.array(validation_target, dtype=numpy.float32)
validation_target = torch.from_numpy(numpy.array(validation_target))
validation_target.requires_grad = False
validation_target = torch.t(validation_target).type(torch.LongTensor).cuda()

# Data Loader for easy mini-batch return in training
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = False, 
                               num_workers = 0, pin_memory=True)
valid_loader = Data.DataLoader(dataset = validation_data, batch_size = BATCH_SIZE, shuffle = False, 
                               num_workers = 0, pin_memory=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_init = nn.Sequential(                   
            nn.Conv2d(L_FIRST, L_SECOND, 3, 1, 0),
            nn.Dropout(p = DROPOUT_INIT),
            nn.ReLU(RELU),                      
        )        
        
        self.conv1 = nn.Sequential(                     
            nn.Conv2d(L_SECOND, L_SECOND, kernel_size=(5, 1), stride=(1, 1), padding = (1,1)),
            nn.Dropout(p = DROPOUT_MIDDLE_1),
            nn.ReLU(RELU),                      
            
            nn.Conv2d(L_SECOND, L_SECOND, kernel_size=(1, 5), stride=(1, 1), padding = (1,1)),
            nn.Dropout(p = DROPOUT_MIDDLE_1),
            nn.ReLU(RELU),                      

            nn.Conv2d(L_SECOND, L_THIRD, kernel_size=(5, 1), stride=(1, 1), padding = (1,1)),     
            nn.Dropout(p = DROPOUT_MIDDLE_1),             
            nn.ReLU(RELU),                      
            
            nn.Conv2d(L_SECOND, L_THIRD, kernel_size=(1, 5), stride=(1, 1), padding = (1,1)),     
            nn.Dropout(p = DROPOUT_MIDDLE_1),             
            nn.ReLU(RELU),                      
        )
        
        self.conv2 = nn.Sequential(                    
            nn.Conv2d(L_SECOND, L_SECOND, 3, 1, 1),
            nn.Dropout(p = DROPOUT_MIDDLE_2),
            nn.ReLU(RELU),                      

            nn.Conv2d(L_SECOND, L_THIRD, 3, 1, 1),     
            nn.Dropout(p = DROPOUT_MIDDLE_2),             
            nn.ReLU(RELU),                      
        )     
        
        self.conv3 = nn.Sequential(                    
            nn.Conv2d(L_SECOND, L_SECOND, 4, 1, 1),
            nn.Dropout(p = DROPOUT_MIDDLE_2),
            nn.ReLU(RELU),                      
            
            nn.Conv2d(L_SECOND, L_SECOND, 4, 1, 1),
            nn.Dropout(p = DROPOUT_MIDDLE_2),
            nn.ReLU(RELU),                      
        ) 
        
        self.conv4 = nn.Sequential(                                          
            nn.Conv2d(L_SECOND, L_SECOND, 2, 1, 0),
            nn.Dropout(p = DROPOUT_MIDDLE_2),
            nn.ReLU(RELU),    
            
            nn.Conv2d(L_SECOND, L_SECOND, 2, 1, 0),
            nn.Dropout(p = DROPOUT_MIDDLE_2),
            nn.ReLU(RELU), 
        ) 

        self.conv5 = nn.Sequential(                                          
            nn.Conv2d(L_SECOND, L_SECOND, 3, 2, 0),
            nn.Dropout(p = DROPOUT_MIDDLE_2),
            nn.ReLU(RELU),                      
        )         
        
        self.drop = nn.Sequential(  
            nn.Dropout(p = DROPOUT_SKIP),
        )                   
        
        self.max_pool = nn.Sequential(  
            nn.MaxPool2d(2, 1),
        )   
              
        self.classifier = nn.Sequential(                  
            nn.Dropout(DROPOUT_CLASSIFIER),
            nn.ReLU(RELU),          
            nn.Linear(L_SEVENTH, 9, bias=False),
        )

    def forward(self, x): 
        x1 = self.conv_init(x)
        x2 = self.conv1(x1) + self.conv2(x1) + self.conv3(x1) + self.conv4(x1) + self.conv5(x1)                 # bring the branches together
        #x3 = self.conv1(x2) + self.conv2(x2) + self.conv3(x2) + self.conv4(x2) + self.conv5(x2) + self.drop(x2)
        #x4 = self.conv1(x3) + self.conv2(x3) + self.conv3(x3) + self.conv4(x3) + self.conv5(x3) + self.drop(x3)
        x = self.max_pool(x2)
        # a linear layer exppects a unidimensional vector of :
        # batch_size x channels x width x height
        x = x.view(x.size(0), -1) 
        output = self.classifier(x)
        return output, x 
    

cnn = CNN()
#print(cnn)  # net architecture
#list(cnn.parameters())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "NVIDIA 1080TI GPUs!")
    cnn = nn.DataParallel(cnn)

cnn = torch.nn.DataParallel(cnn, device_ids=[0]).cuda()
cnn.to(device)

# load a pre-trained network
#print("Loaded a model with low Validation Loss!")
#cnn.load_state_dict(torch.load('class2.pt'))  

train_data, train_target = train_data.to(device), train_target.to(device)
validation_data, validation_target = validation_data.to(device), validation_target.to(device)
test_data, test_target = test_data.to(device), test_target.to(device)

#loss_func = nn.CrossEntropyLoss()
#loss_func = nn.NLLLoss()
loss_func = nn.MultiMarginLoss()

#optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION) # 
#optimizer = torch.optim.Adadelta(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION)
optimizer = torch.optim.Adagrad(cnn.parameters(), lr=LR, weight_decay=REGULARIZATION) 
#optimizer = optim.SGD(cnn.parameters(), lr=LR, momentum=MM, weight_decay=REGULARIZATION)


def train():
    cnn.train()
    for step, data in enumerate(train_loader,0):
        optimizer.zero_grad()                         # clear the gradients of all optimized variables            
        output = cnn(train_data)[0]                   # forward pass: compute predicted outputs by passing inputs to the model        
        print('output: ',output)
        print(len(output))
        print(len(train_target[0]))
        print('train target: ',train_target[0])
        loss = loss_func(output, train_target[0]) 
        
        train_losses.append(loss.item())              # record training loss             
        last_training_loss = loss.item()
               
        loss.backward()                               # backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()                              # perform a single optimization step (parameter update)

        
def validation(cnn, valid_loader, loss_func):
    valid_loss = 0
    accuracy = 0

    for inputs, classes in enumerate(valid_loader,0):
        output = cnn(validation_data)[0]            # forward pass: compute predicted outputs by passing inputs to the model
        valid_loss += loss_func(output, validation_target[0]).item()
          
        #valid_losses.append(test_loss.item())          # record validation loss
        #valid_losses.append(valid_loss)                 # record validation loss
        
        ps = torch.exp(output)
        equality = (validation_target[0].data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy           


test_loss = min_val_loss;
# training 
for epoch in range(EPOCHS):
    loss = None
        
    train()   
    cnn.eval()                 # switch to evaluation (no change) mode
    with torch.no_grad():
        valid_loss, accuracy = validation(cnn, valid_loader, loss_func)

    test_output, last_layer = cnn(train_data)
    pred_y = torch.max(test_output, 1)[1]
    print("---------------------")
    print("Epoch: {}/{}.. ".format(epoch+1, EPOCHS))
    print("Training Accuracy  : {:.2f} ".format(torch.sum(pred_y == train_target[0]).type(torch.FloatTensor) / float(train_target.size(1))),
        "Validation Accuracy: {:.2f}".format(accuracy/len(valid_loader)))
    
    # print training/validation statistics 
    train_loss = train_losses[-1]
    
    epoch_len = len(str(EPOCHS))  
   
    print_msg = (f' Training Loss  : {train_loss:.2f} ' +
                 f'  Validation Loss: {valid_loss:.2f}')
        
    print(print_msg)
       
    # clear lists to track next epoch
    train_losses = []
    valid_losses = []
        
    # early_stopping needs the validation loss to check if it has decresed, 
    # and if it has, it will make a checkpoint of the current model
    early_stopping(valid_loss, cnn)
    reduced_rate(valid_loss, cnn)
    
    if reduced_rate.early_stop2:
        print("Reducing Learning Rate")
        LR = LR / 2
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

# load the last checkpoint with the best model
print("Loaded the model with the lowest Validation Loss!")
cnn.load_state_dict(torch.load('checkpoint.pt'))        
    
    
#test           
with torch.no_grad():
    outputs = cnn(test_data)[0]
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.to(device)

    c = (predicted == test_target[0]).squeeze()
    c = c.to(device)
    
    
for j in range (TEST_SIZE):
    if (c[j] == 1): 
        label2[0,predicted[j]] += 1 
            
#print('Correct classif. in each class  : ',label2)
#print('Total number of pixels per class: ',label1)            
percent = (torch.sum(c).item()/TEST_SIZE)
print('Correct Classification (Percent): %.2f' % percent)
print('Results by class: ',label2/label1)

end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print('Total execution time (minutes): ',start.elapsed_time(end)/60000)

torch.cuda.empty_cache()

