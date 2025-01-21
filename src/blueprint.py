"""This file holds the blue print of the AMT model"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import librosa
import json

import csv
import pandas as pd
import pretty_midi as pm
import mido
import IPython.display as ipd
import torch
from scipy.signal import resample
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import math
import time
import pickle




class PianoNoteModel(nn.Module):
    def __init__(self, num_mel_bins=88, mel_temporal_length=89, num_frame_output=22, output_size=(88, 3)):
        """The default parameter is approximated for 1 seconds of audio data, regarding to the temporal_length"""
        super(PianoNoteModel, self).__init__()

        self.num_mel_bins = num_mel_bins
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
        # Compute flattened size based on input dimensions after pooling
        # Assuming input shape is (batch_size, 1, num_mel_bins, num_frames)
        pooled_mel_bins = num_mel_bins // 2  # Adjust based on pooling
        pooled_temporal_length = mel_temporal_length // 2     # Adjust based on pooling

        
        flattened_size = pooled_mel_bins * pooled_temporal_length * 64  # Based on conv2 output channels
        
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_frame_output * output_size[0] * output_size[1])  # Predict for each frame
        
    def forward(self, x):
        # Input shape: (batch_size, 1, num_mel_bins, num_frames)
        
        # Convolutional layers
        
        x = F.relu(self.conv1(x))

       

        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.dropout(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten except batch dimension
       
        # Fully connected layers
        x = F.relu(self.fc1(x))
       
        x = self.fc2(x)
        
        # Reshape to output dimensions: (batch_size, num_frames, 88, 3)
        x = x.view(x.size(0), -1, 88, 3)
        
        return x
    
class HCQTModel(nn.Module):
    def __init__(self, num_harmonics=3, num_cqt_bins=88, cqt_temporal_length= 89, num_frame_output=22, output_size=(88,3)):
        super(HCQTModel, self).__init__()
        self.num_harmonics = num_harmonics
        self.num_cqt_bins=num_cqt_bins
        self.output_size = output_size

         # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=num_harmonics, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
        # Compute flattened size after pooling
        pooled_cqt_bins = num_cqt_bins // 2  # Adjust based on pooling
        pooled_temporal_length = cqt_temporal_length // 2  # Adjust based on pooling
        
        flattened_size = pooled_cqt_bins * pooled_temporal_length * 64  # Based on conv2 output channels
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, num_frame_output * output_size[0] * output_size[1])  # Predict for each frame
        
    def forward(self, x):
        """
        Forward pass through the network.
        """
        
        
        # Input shape: (batch_size, 1, num_mel_bins, num_frames)
        
        # Convolutional layers
        
        x = F.relu(self.conv1(x))

       

        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.dropout(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten except batch dimension
       
        # Fully connected layers
        x = F.relu(self.fc1(x))
       
        x = self.fc2(x)
        
        # Reshape to output dimensions: (batch_size, num_frames, 88, 3)
        x = x.view(x.size(0), -1, 88, 3)
        
        return x


# Custom loss function, for mutipurpose loss function in the output layer
class MultiTaskLoss(nn.Module):
    def __init__(self, classification_weight=1.0, regression_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.classification_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.regression_loss = nn.MSELoss()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight

    def forward(self, classification_output, classification_target, 
                regression_output, regression_target, ratio):
        # Compute classification loss
        loss_per_element  = self.classification_loss(classification_output, classification_target)

        pressed_key_mask = classification_target == 1
        unpressed_key_mask = classification_target == 0

        pressed_key_label = classification_target[pressed_key_mask]
        unpressed_key_label = classification_target[unpressed_key_mask]


        pressed_keynumber = pressed_key_label.numel()
        unpressed_keynumber = unpressed_key_label.numel()


        # Compute the scoring of pressedkey and unpressed key with the feedback, my empirical
        epsilon = 1e-8  # Small constant to avoid instability
        if (pressed_keynumber == 0):
            w_pressed = 0
        else: 
            w_pressed = ((unpressed_keynumber/(pressed_keynumber+ epsilon))**(ratio/2)) #* feed_back[0]
            
        if (unpressed_keynumber == 0):
            w_unpressed = 0
        else: 
            w_unpressed= ((pressed_keynumber/(unpressed_keynumber+ epsilon))**(ratio/2)) #* feed_back[1]


        #total =  pressed_keynumber + unpressed_keynumber
        #w_pressed = unpressed_keynumber /total
        #w_unpressed = pressed_keynumber / total


        #simple pitch empirical
        #total =  pressed_keynumber + unpressed_keynumber
        #w_pressed = 0.95 
        #w_unpressed = 0.05
        #---


        weight_matrix = torch.full_like(classification_target, w_unpressed)
        weight_matrix[pressed_key_mask] = w_pressed
        
        weighted_class_loss_mat = loss_per_element * weight_matrix
        
        weighted_class_loss = weighted_class_loss_mat.sum()
        
        
        """# Computing the current accuracy of pressed and non pressed keys
        predicted_class = (classification_output > 0.5).float()
        correct_pressed = (predicted_class[pressed_key_mask] == classification_target[pressed_key_mask]).sum().item()
        correct_unpressed = (predicted_class[unpressed_key_mask] == classification_target[unpressed_key_mask]).sum().item()
        total_pressed = pressed_key_mask.sum().item()
        total_unpressed = unpressed_key_mask.sum().item()



        # update the feedback based on the accuracy
        epsilon = 1e-6  # Small constant to avoid instability
        correct_pressed_ratio = max(correct_pressed / total_pressed, epsilon)
        correct_unpressed_ratio = max(correct_unpressed / total_unpressed, epsilon)

        feed_back[0] = 1 / correct_pressed_ratio
        feed_back[1] = 1 / correct_unpressed_ratio"""

        # Compute regression loss
        reg_loss = self.regression_loss(regression_output, regression_target)
        
        # Combine with weights
        total_loss = self.classification_weight * weighted_class_loss + self.regression_weight * reg_loss
        
        return total_loss
    
    def set_classification_balancer(self, weight):
        self.classification_loss.weight = weight
    













