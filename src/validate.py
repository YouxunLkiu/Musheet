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

from src.preprocess import load_index_from_csv
import random
from src.blueprint import HCQTModel, PianoNoteModel, MultiTaskLoss
from src.training import load_model, generate_data_label, TensorDataset, DataLoader



def validation_accuracy_check(id, index, noised=True, num_frame=22, segment_jump = 0.5, frame_length = 2048, hop_length = 512, batch_size=16, CQT=True):
    """This function returns the accuracy of the model predicting the song at index
        arguments:
            id: trained model id
            index: index of the song.
    """
    if CQT:
        model = HCQTModel()
    else:
        model = PianoNoteModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #Letting the classification utility be more potent than the regression utility.
    criterion = MultiTaskLoss(classification_weight=1.5, regression_weight=0.7)
 
    
    model, optimizer = load_model(model, optimizer, id)
   

    datas, labels = generate_data_label(index, num_frame=num_frame, segment_jump=segment_jump, frame_length=frame_length, hop_length=hop_length)
    

    datas = torch.tensor(datas, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(datas, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    classification_correct = 0
    classification_true_correct = 0
    regression_error = 0

    true_classified = 0
    total_classifed = 0
    num_classified = 0
    
    with torch.no_grad():
        
        for val_inputs, val_labels in dataloader:
            outputs = model(val_inputs)

            #Output separation
            classification_outputs = outputs[:,:,:, 0:1].squeeze(-1)
            regression_outputs = outputs[:, :, :, 1:3]

            #Label separation
            classification_labels = val_labels[:, :, :, 0:1].squeeze(-1)
            regression_labels = val_labels[:, :, :, 1:3]
            
            #Calculate the classification accuracy of a frame
            predicted_classes = (classification_outputs > 0.5).float()
            classification_correct += (predicted_classes == classification_labels).sum().item() / model.num_mel_bins #this should be the accuracy of a frame


            #Calculate the classification accuracy of pressed key
            pressed_key_mask = classification_labels == 1
            pressed_key_label = classification_labels[pressed_key_mask]
            pressed_key_prediction = predicted_classes[pressed_key_mask]
            true_classified += pressed_key_label.numel()
            total_classifed += classification_labels.numel()
            classification_true_correct += (pressed_key_label == pressed_key_prediction).sum().item()
            

            regression_error += ((regression_outputs - regression_labels) ** 2).mean().item()
            
            num_classified += val_labels.shape[0] * val_labels.shape[1] # counting number of frames have been classified i.e. batch_number * number of frame per segment
            

    if true_classified > 0 :
        true_accuracy = classification_true_correct/true_classified
    else:
        true_accuracy = 0
    
 
    return true_accuracy, classification_correct/ num_classified, 



import numpy as np




"""Validating N random songs from the validation set and returns the average of their accuracy in frames and pressed_notes"""
def validation_N_songs(id, N, cache = False, mel= False, CQT =True, model=None, optimizer=None, noised=True, num_frame=22, segment_jump = 0.5, frame_length = 2048, hop_length = 512, batch_size=16):
    if model == None and optimizer == None:
        if CQT:
            model = HCQTModel()
        else:
            model = PianoNoteModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        model, optimizer = load_model(model, optimizer, id)
    all_indices = load_index_from_csv("traindata/maestro-v3.0.0-midi/maestro-v3.0.0/validation_indicies.csv")
    indices_subset = random.sample(all_indices,N)

    classification_correct = 0
    classification_true_correct = 0
    regression_error = 0

    true_classified = 0
    total_classifed = 0
    num_classified = 0


    ##Use the bellow to calculate the f1 score
    true_positive = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for index in indices_subset:
            datas, labels = generate_data_label(index, num_frame=num_frame, segment_jump=segment_jump, frame_length=frame_length, hop_length=hop_length, HCQT=CQT, mel = mel)
            datas = torch.tensor(datas, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32)
            dataset = TensorDataset(datas, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for val_inputs, val_labels in dataloader:
                outputs = model(val_inputs)

                #Output separation
                classification_outputs = outputs[:,:,:, 0:1].squeeze(-1)
                regression_outputs = outputs[:, :, :, 1:3]

                #Label separation
                classification_labels = val_labels[:, :, :, 0:1].squeeze(-1)
                regression_labels = val_labels[:, :, :, 1:3]
                
                #Calculate the classification accuracy of a frame
                predicted_classes = (classification_outputs > 0.5).float()
                if CQT:
                    bin = model.num_cqt_bins
                else:
                    bin = model.num_mel_bins
                classification_correct += (predicted_classes == classification_labels).sum().item()/bin  #this should be the accuracy of a frame

                true_positive += torch.sum((classification_labels == 1) & (predicted_classes == 1.0)).item()
                false_positive += torch.sum((classification_labels == 0) & (predicted_classes == 1.0)).item()
                false_negative += torch.sum((classification_labels == 1) & (predicted_classes == 0)).item()

                
                #Calculate the classification accuracy of pressed key
                pressed_key_mask = classification_labels == 1
                pressed_key_label = classification_labels[pressed_key_mask]
                pressed_key_prediction = predicted_classes[pressed_key_mask]
                true_classified += pressed_key_label.numel() 
                total_classifed += classification_labels.numel()
                classification_true_correct += (pressed_key_label == pressed_key_prediction).sum().item()
                

                regression_error += ((regression_outputs - regression_labels) ** 2).mean().item()
                
                num_classified += val_labels.shape[0] * val_labels.shape[1]
    if true_classified > 0 :
        true_accuracy = classification_true_correct/true_classified
    else:
        true_accuracy = 0
    print(f"True accuracy{true_accuracy}, Frame accuracy {classification_correct/num_classified}.")
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    

    print(f"F1 Score is {f1_score}")
    return true_accuracy, classification_correct/ num_classified, true_classified/total_classifed, f1_score



