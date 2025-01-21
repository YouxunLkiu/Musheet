import preprocess
import blueprint

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


from torch.utils.data import DataLoader, TensorDataset
import tqdm
import random
import re
import os

## MODEL Saver
def save_model(model, optimizer, id, path = "models/"):
    """ Saving the model after each training/testing before each training progress"""
    actualPath = path + f"{id}_piano_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, actualPath)
    
    
## MODEL loader
def load_model(model, optimizer, id, path = "models/"):
    actualPath = path + f"{id}_piano_model.pth"
    checkpoint = torch.load(actualPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


"""This function generates the matrix label to aligne the mal_spectral gram
each matrix should consired num_frame,
poping fram_jumping after the matrix is being yield

jumping time should be
frame_jumping * (frame_len /sr)
"""
def label_generator(num_frame, frame_jumping, jump_len:int, frame_len: int, sr:int, midi):
    midi_frame_gen = PeekableGenerator(frame_aligning_midi(0, frame_len, frame_len, sr, midi)) ## Note, while using the architechture of mel_spectrogram, the we don't need to consider the jump offset in the midi side
    label = []
    counter = 0
    
    
    concur_time = 0
    concur_time_end = 0
    frame_time = frame_len /sr

    
    while midi_frame_gen.has_next():
        if (len(label) < num_frame):
            label.append(list(midi_frame_gen.__next__()))
            concur_time_end += frame_time
            
        else:
            counter+=1
            yield label
            for i in range(int(frame_jumping)):
                label.pop(0)
                concur_time += frame_time
                


"""This function tries to mimic the decayed velocity miniking the sound at which a piano has been decayed"""
def velocity_decay_sustain (velocity, onset, at_time):
    if (at_time - onset < 0.2):
        return velocity
    else:
        return math.exp((onset - at_time) * 0.6) * velocity


"""formating the label of list, in to a matrix of 88 * 3 matrix.
    each row represent a strikable key, 
    column 1 (being stricked) : 0, 1 (classification purpose)
    column 2 (onset timer) : the set of positive interger that is less than onset. (Regression purpose)
    column 3 (velocity of which is being stricked) : the set of positive integer that is less than onset. (Regression purpose) 
        note for the velocity of the piano key will be approximately alingned with a decay parameter"""
def label_formater(label, frameonset):
    ret_label = np.zeros((len(label), 88,3))
    for i in range(len(label)) :
        for key_obj in label[i]:
            pitch = key_obj['pitch'] - 21
            ret_label[i][pitch][0] = 1
            ret_label[i][pitch][1] = key_obj['start'] - frameonset if key_obj['start'] > frameonset else 0
            ret_label[i][pitch][2] = velocity_decay_sustain(key_obj['velocity'],  key_obj['start'], frameonset) 
    
   

    return ret_label





def construct_input(spectrogram, x, y):
    """This function assure the input for the training will retain the dimension in the case of track is ending."""
    if spectrogram.shape == (x,y):
        return spectrogram
    else: 
        ret = np.zeros((x,y))
        
       
        for i in range(spectrogram.shape[0]):
            ret[i][:spectrogram.shape[1]] = spectrogram[i][:spectrogram.shape[1]]
        return ret
    
def construct_input_hcqt(spectrogram, x, y, harmonic=3):
    if (spectrogram.shape == (harmonic, x, y)):
        return spectrogram
    else :
        ret = np.zeros((harmonic, x, y))
        ret [:harmonic, :x, :y] = hcqt[:harmonic, :x, :y]
        return ret


def train_segment(wav, sr, length=1, hop=0.5):
    ret = []
    max = len(wav)/sr
    begin = 0
    while (begin < max):
        adding = length if begin + length < max else max - begin
        ret.append((begin, begin + adding))
        begin += hop

    return ret
    

def label_buffer(num_frame, label):
    """This function is used at the end of the piece, when the label doesn't extend as long of th piece"""
    if len(label) < num_frame:
        for i in range(num_frame- len(label)):
            label.append([])
    return label




def save_to_file(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_from_file(filename):
    if not os.path.exists(filename):
        return None  # Or a default value like {} or [] based on your use case
    with open(filename, "rb") as f:
        return pickle.load(f)



""" This function will generate all the mel_spectrogram and label pair for each audio segment of the song at index"""
def generate_data_label(index, cache = False, noised=True, num_frame=22, segment_jump = 0.5, frame_length = 2048, hop_length = 512, mel = False, HCQT = True):
    # fetching infomation
    (wav, sr), midi = load_wav_midi_pair(index)
    if mel and HCQT:
        print("not supporting both spectral gram")
        return None
    if noised:
        wav = add_gaussian_noise(wav)

    #Load data and label if they have been generated already
    data_path = f"traindata/preprocessed/data/{index}.json"
    label_path = f"traindata/preprocessed/label/{index}.json"

    loaded_data = load_from_file(data_path)
    loaded_label = load_from_file(label_path)
    if type(loaded_data) != type(None) and type(loaded_label) != type(None):
        return loaded_data, loaded_label


    #asuming this is the first time generating the data label pair 

    frame_time =  frame_length/sr
    quick_sample = audio_segment_between(0, num_frame*frame_time, wav, sr)
    x, y = extract_mel_spectrogram(quick_sample, sr).shape
    segments = train_segment(wav, sr, length = num_frame * frame_time, hop= num_frame * frame_time/2)
    
    label_gen = PeekableGenerator(label_generator(num_frame, num_frame*segment_jump, hop_length, frame_length, sr, midi))
    
    input_datas = []
    labels = []
    
  
    for beg, end in segments:
        
        if (label_gen.has_next() is False) :
            break
        
        audio = audio_segment_between(beg, end, wav, sr)
        if mel: 
            mel_spectrogram = extract_mel_spectrogram(audio, sr, hop_length= hop_length, n_fft=frame_length)
            input_data = construct_input(mel_spectrogram, x,y)
            input_data = np.array(input_data)
            input_data = np.expand_dims(input_data, axis=0)
            input_datas.append(input_data)
            

        if HCQT:
            HCQT_spectrogram = extract_hcqt_spectrogram(audio, sr, hop_length=hop_length)
            input_data = construct_input_hcqt(HCQT_spectrogram, x,y, 3)
            input_data = np.array(input_data)
            input_datas.append(input_data)
        
        

        
        label = label_buffer(num_frame, label_gen.__next__())
        label = label_formater(label, beg)        
        labels.append(label)
    if cache :
        save_to_file(np.array(input_datas), data_path)
        save_to_file(np.array(labels), label_path) 
    
    return np.array(input_datas), np.array(labels)
   


def one_pass_song_train(id, index, ratio, lr = 0.001, cache = False, noised=True, num_frame=22, segment_jump = 0.5, frame_length = 2048
                        , hop_length = 512, batch_size=16, mel = False, CQT = True ):
    #load model
    """The training is based on how many frame should be trained at a time,
        the default setting is suited for the expriment set up above,
        num_frame is tried to aligned it to approx 1 second of the sample
        segment_jumping would be trying to get 50% of the sample audio
        The dimension of the input is:
        batchsize * channel * 
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if CQT:
        model = HCQTModel()
    else:
        model = PianoNoteModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #Letting the classification utility be more potent than the regression utility.
    criterion = MultiTaskLoss(classification_weight=1.7, regression_weight=0.5)
  
    model, optimizer = load_model(model, optimizer, id)
    #override the optimizer with our lr.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # generate all data
    datas, labels = generate_data_label(index, cache = False, noised=noised, num_frame=num_frame, segment_jump=segment_jump,
                                         frame_length=frame_length, hop_length=hop_length, mel=mel, HCQT=CQT)
    datas = torch.tensor(datas, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(datas, labels)
    # Let model be in training mode
    model.train()
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle =True)
    # Initialize the feedback value before training to 
    

    lossval = 0.0
    #for _, (batch_inputs, batch_labels) in progress_bar:
    for batch_inputs, batch_labels in dataloader:
        optimizer.zero_grad()

        outputs = model(batch_inputs)

        #  classification training
        classification_output = outputs[..., 0:1]
        classification_target = batch_labels[..., 0:1]
        
        #  regression training
        regression_output = outputs[..., 1:3]
        regression_target = batch_labels[...,1:3]
        loss = criterion.forward(classification_output, classification_target, regression_output, regression_target, ratio)
    
        

        loss.backward()
        optimizer.step()

        lossval+= loss.item()

        
    #save model
    save_model(model, optimizer, id)
    return lossval

def save_training_message_log(id, log):
    # Open the file in write mode and write the string
    path = "models/"+ id + "_training_log.txt"
    with open(path, "w") as file:
        file.write(log)

def extract_ratios_from_log_file(id):
    path = "models/"+ id + "_training_log.txt"
    # Open the file and read its contents
    with open(path, 'r') as file:
        text = file.read()
    # Regular expression to match "Ratio of pressed is in data is <value>"
    pattern = r"The ratio was trained with (\d+\.\d+)"
 
    # Find all matching ratios in the text
    ratios = re.findall(pattern, text)
    
    # Convert each ratio to a float for easier processing (optional)
    return [float(ratio) for ratio in ratios]

def extract_fc_from_log_file(id):
    path = "models/"+ id + "_training_log.txt"
    # Open the file and read its contents
    with open(path, 'r') as file:
        text = file.read()
    # Regular expression to match "Ratio of pressed is in data is <value>"
    pattern = r"Frame_accuracy: (\d+\.\d+)"
 
    # Find all matching ratios in the text
    ratios = re.findall(pattern, text)
    
    # Convert each ratio to a float for easier processing (optional)
    return [float(ratio) for ratio in ratios]

def extract_pc_from_log_file(id):
    path = "models/"+ id + "_training_log.txt"
    # Open the file and read its contents
    with open(path, 'r') as file:
        text = file.read()
    # Regular expression to match "Ratio of pressed is in data is <value>"
    pattern = r"pressed_corrected: (\d+\.\d+)"
 
    # Find all matching ratios in the text
    ratios = re.findall(pattern, text)
    
    # Convert each ratio to a float for easier processing (optional)
    return [float(ratio) for ratio in ratios]

def loss(pc, fc):
    return abs(1-pc) + abs(1-fc)

def train_N_song_on_epoch (id, epoch, N, back_up, lr = 0.001, cache = False, ratio_update = True, ratio_load_prev = True, ratio=1.5, ratio_dummy = 1.51, pc = 0.7928640553348053,
                            fc = 0.5865201712217555, repetition = 2, noised=True, CQT = True, Mel= False):
    
    """prev_ratio = extract_ratios_from_log_file(id)[-1]
    if (prev_ratio != None and ratio_load_prev):
        prev = prev_ratio"""
    
    tobe_done = load_progress_index_from_csv(epoch, id)

    N = min(N, len(tobe_done))
    ## indices to be trained for the current run
    parse_in = [tobe_done.pop() for _ in range(N)]

    random.shuffle(parse_in)

    progress = tqdm.tqdm(parse_in)
    counter = 0
    trained = []
    ## Passing in each index to train on one_pass_song_train
    training_message = ""
    for index in progress:
        
        for rep in range(repetition):
            loss = one_pass_song_train(id, index, ratio, noised=noised, lr = lr)
            training_message += f"loss value at {index} on {rep} is {loss}" + '\n' 
            
        trained.append(index)
        counter += 1
        """if (counter % 7 == 0):
            new_pc, new_fc, _ = validation_N_songs(id, 5)
            
            training_message += '\n' + f"The ratio was trained with {ratio}. Trained {len(trained)} songs." + '\n'
            training_message += f"pressed_corrected: {new_pc}. Frame_accuracy: {new_fc}." + '\n'
            dl = loss(new_pc, new_fc) - loss(pc, fc)
            dr = ratio - ratio_dummy + 1e-5
            pc = new_pc
            fc = new_fc
            if (ratio_update):
                ratio_dummy = ratio
                ratio = ratio + 0.1*(dl/dr)"""
        #save this to a messaage file
        
        #save_training_message_log(id, training_message)
    save_training_message_log(id, training_message)
    

    new_pc, new_fc, _ = validation_N_songs(id, 10)
            
    training_message += '\n' + f"The ratio was trained with {ratio}. Trained {len(trained)} songs." + '\n'
    training_message += f"pressed_corrected: {new_pc}. Frame_accuracy: {new_fc}." + '\n'
    save_training_message_log(id, training_message)
    
    if (len(tobe_done) == 0):
        print(f"Epoch {epoch} training complete.")
        save_progress_index_to_csv(tobe_done, epoch, id)
    else :
        
        print(f"Epoch {epoch} have trained {N} songs, left over songs for current epoch will be saved. Need to train {len(tobe_done)}.")
        save_progress_index_to_csv(tobe_done, epoch, id)
    actualPath = f"models/{id}_piano_model.pth"
    print(f"Model saved to {actualPath}")

    if CQT:
        model = HCQTModel()
    else:
        model = PianoNoteModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #Letting the classification utility be more potent than the regression utility.
  
    model, optimizer = load_model(model, optimizer, id)

    back_up_id = "" + id + back_up
    save_model(model, optimizer, back_up_id)
    save_training_message_log(back_up_id, training_message)
    print(f"back_up_id {id} + {back_up}")

def model_roll_back(fromID, BackupID):
    model = PianoNoteModel(output_size=(88, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model, optimizer = load_model(model, optimizer, BackupID)
    save_model(model, optimizer, fromID)
