#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn

class convLSTMNet(nn.Module):
    def __init__(self, batch_size):
        super().__init__() # just run the init of parent class (nn.Module)

        self.batch_size = batch_size
        self.E_dp_prob = 0.5
        self.fc_dp_prob = 0.5
        
        self.num_conv_kernels_1 = 10
        self.num_conv_kernels_2 = 20   
        self.num_conv_kernels_3 = 30   

        self.CNN_model_init()

        x = torch.randn(self.batch_size, 64, 64).view(-1, 1, 64, 64)
        self.eeg_conv_shape = None
        self.CNN_model_convs(x)   # eeg_conv_shape gets updated 

        self.lstm_hidden_size = 48
        use_bidirectional = True

        if True == use_bidirectional:
            self.num_direction = 2
            self.direction_scale = 0.5
        else:
            self.num_direction = 1
            self.direction_scale = 1 
            
        self.lstm1 = nn.LSTM((self.eeg_conv_shape[1]*self.eeg_conv_shape[3]*3), int(self.lstm_hidden_size*self.direction_scale), 
                             bidirectional=use_bidirectional, batch_first=True)            

        tmp = self.lstm_hidden_size*self.eeg_conv_shape[2]      
        self.fc1 = nn.Linear(tmp, 128) 
        self.fc1_dp = nn.Dropout(p=self.fc_dp_prob)

        self.fc2 = nn.Linear(128, 128) 
        self.fc2_dp = nn.Dropout(p=self.fc_dp_prob)

        self.fc3 = nn.Linear(128, 32) 
        self.fc3_dp = nn.Dropout(p=self.fc_dp_prob)

        self.fc4 = nn.Linear(32, 1)        

    def CNN_model_init(self):

        self.conv1 = nn.Conv2d(1, self.num_conv_kernels_1, kernel_size=(8,4),padding ='same')
        self.mPool1 = nn.MaxPool2d((2,2))
        self.conv1_bn = nn.BatchNorm2d(self.num_conv_kernels_1)
        self.conv1_dp = nn.Dropout(p=self.E_dp_prob) 
 
        self.conv2 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_2, kernel_size=(4,16), padding='same')
        self.mPool2 = nn.MaxPool2d((2,8))
        self.conv2_bn = nn.BatchNorm2d(self.num_conv_kernels_2)
        self.conv2_dp = nn.Dropout(p=self.E_dp_prob) 

        self.conv3 = nn.Conv2d(self.num_conv_kernels_2, self.num_conv_kernels_3, kernel_size=(2,8), padding='same')
        self.mPool3 = nn.MaxPool2d((2,4))
        self.conv3_bn = nn.BatchNorm2d(self.num_conv_kernels_3)   
        self.conv3_dp = nn.Dropout(p=self.E_dp_prob) 

    def CNN_model_convs(self, x):
        # order: CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x = self.conv1_bn(self.mPool1(self.conv1(x)))
        x = self.conv1_dp(F.relu(x))

        x = self.conv2_bn(self.mPool2(self.conv2(x)))
        x = self.conv2_dp(F.relu(x))

        x = self.conv3_bn(self.mPool3(self.conv3(x)))
        x = self.conv3_dp(F.relu(x))      

        if self.eeg_conv_shape is None:
            self.eeg_conv_shape = x.shape
            print(self.eeg_conv_shape)
        return x

    
    def forward(self, eeg_x1, eeg_x2, eeg_x3):
        
        eeg_x1 = self.CNN_model_convs(eeg_x1)
        # for the lstm input, first dim after batch size should be seq_len 
        eeg_x1 = eeg_x1.view(-1, self.eeg_conv_shape[2], self.eeg_conv_shape[3]*self.eeg_conv_shape[1])
        eeg_x2 = self.CNN_model_convs(eeg_x2)
        eeg_x2 = eeg_x2.view(-1, self.eeg_conv_shape[2], self.eeg_conv_shape[3]*self.eeg_conv_shape[1])      

        eeg_x3 = self.CNN_model_convs(eeg_x3)
        eeg_x3 = eeg_x3.view(-1, self.eeg_conv_shape[2], self.eeg_conv_shape[3]*self.eeg_conv_shape[1])
        
        x = torch.cat([eeg_x1, eeg_x2, eeg_x3], dim=2)

        # Input = batch, seq_len, input_size (self.eeg_conv_shape[0], self.eeg_conv_shape[2]*self.eeg_conv_shape[3], self.eeg_conv_shape[1]) 
        # Output = batch, seq_len, self.lstm_hidden_size
        # hidden state: num_layers * num_directions, batch, self.lstm_hidden_size*self.direction_scale  (read my notes from book)
        # cell state: same as hidden state
        
        x, (hidden_state, cell_state) = self.lstm1(x)
        x = x.reshape(-1, self.lstm_hidden_size*self.eeg_conv_shape[2])
        # order: CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x = self.fc1_dp(F.relu(self.fc1(x)))
        x = self.fc2_dp(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

