""" This file implements all necessary functions that are use for preprocessing datas.

    file loadings from the data directories




"""

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


### making the random seed 
np.random.seed(40)



labels_file_path = "traindata/maestro-v3.0.0-midi/maestro-v3.0.0/maestro-v3.0.0.json"
with open(labels_file_path, 'r') as file:
    data = json.load(file)

all_sets = {}
all_sets['train'] = []
all_sets['validation'] = []
all_sets['test'] = []

def sortingsets (data, allsets):
    for key in data:
       
        if data[key] == 'train':
            all_sets['train'].append(key)
        elif data[key] == 'validation':
            all_sets['validation'].append(key)
        else:
            all_sets['test'].append(key)

def save_index_to_csv(all_sets):
    for key in all_sets:
        path = f"traindata/maestro-v3.0.0-midi/maestro-v3.0.0/{key}_indicies.csv"
        df = pd.DataFrame({'Index': all_sets[key]})
        df.to_csv(path, index=False)

def load_index_from_csv(path):
    indices = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            indices.append(int(row[0]))
    return indices

def save_progress_index_to_csv(indices, epoch, id):
    path = f"models/training_index{epoch}_for_{id}.csv"
    df = pd.DataFrame({'Index': indices})
    df.to_csv(path, index=False)

def load_progress_index_from_csv(epoch, id):
    path = f"models/training_index{epoch}_for_{id}.csv"
    indices = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            indices.append(int(row[0]))
    return indices

sortingsets(data['split'], all_sets)
save_index_to_csv(all_sets)







## Randomizing the data set index for training purposes
def randomizeing(data_set):
    ds = np.array(data_set)
    np.random.shuffle(ds)
    return ds

## Select n indices from the givien data set
def select_N_randomized_from_set(n, data_set):
    nparry = randomizeing(data_set)
    return nparry[:n]


## ----- ----- ---------- loading function ------------------------------ ##

## function loading in the wav function
def load_wav_from_index(index):
    labels_file_path = "traindata/maestro-v3.0.0-midi/maestro-v3.0.0/maestro-v3.0.0.json"
    with open(labels_file_path, 'r') as file:
        data = json.load(file)
    wav_path = "traindata/maestro-v3.0.0/maestro-v3.0.0/" + data['audio_filename'][str(index)]
    
    return librosa.load(wav_path, sr=None)

## function loading in the midi function
def load_midi_from_index(index):
    labels_file_path = "traindata/maestro-v3.0.0-midi/maestro-v3.0.0/maestro-v3.0.0.json"
    with open(labels_file_path, 'r') as file:
        data = json.load(file)
    midi_path = "traindata/maestro-v3.0.0-midi/maestro-v3.0.0/" + data['midi_filename'][str(index)]
    return pm.PrettyMIDI(midi_path)

## ----- ----- -------- Path showing function-------------------------------- ##
    
## showing the file path audio of the wave
def show_wav_path(index):
    labels_file_path = "traindata/maestro-v3.0.0-midi/maestro-v3.0.0/maestro-v3.0.0.json"
    with open(labels_file_path, 'r') as file:
        data = json.load(file)
    wav_path = "traindata/maestro-v3.0.0/maestro-v3.0.0/" + data['audio_filename'][str(index)]
    return wav_path

## Showing the file path of the midi file of data[index]
def show_midi_path(index):
    labels_file_path = "traindata/maestro-v3.0.0-midi/maestro-v3.0.0/maestro-v3.0.0.json"
    with open(labels_file_path, 'r') as file:
        data = json.load(file)
    midi_path = "traindata/maestro-v3.0.0-midi/maestro-v3.0.0/" + data['midi_filename'][str(index)]
    return midi_path

## loading in the wav and midi pair
def load_wav_midi_pair(index): ## (wav, midi)
    return load_wav_from_index(index), load_midi_from_index(index)







""" This function extracts all played notes in the midi Object, which it will be futher trained with the aligne ed wave object
    input: pm object
    It is good for debugging and seeing the midi object
"""
def extract_midi_notes(midi, format = "piano_roll", fs=1000):
    if (format == "piano_roll"):
        return extract_midi_notes_piano_roll(midi, fs)
    else:
        return extract_midi_notes_raw(midi)



def extract_midi_notes_raw(midi):
    """This function extract from the raw midi instrument for the midi information"""
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.start + note.get_duration(),
                'velocity': note.velocity
            })
    #preprocessing the sort
    notes.sort(key=(lambda x: x['start']))
    return notes


def extract_midi_notes_piano_roll(midi, fs):
    """This function extract from the piano row of the instruments for the midi information"""
    notes = []
    piano_roll = midi.get_piano_roll(fs=fs)
    time_step = 1 / fs  # Duration of each time frame in seconds

    # Iterate over each pitch (row in the piano roll)
    for pitch, row in enumerate(piano_roll):
        active = False
        note_start = None
        velocity = 0

        for t, value in enumerate(row):
            if value > 0 and not active:  # Note starts
                active = True
                note_start = t * time_step
                velocity = value
            elif value == 0 and active:  # Note ends
                active = False
                note_end = t * time_step
                notes.append({
                    'pitch': pitch,
                    'start': note_start,
                    'end': note_end,
                    'velocity': velocity
                })

        # If a note is still active at the end of the piano roll
        if active:
            note_end = len(row) * time_step
            notes.append({
                'pitch': pitch,
                'start': note_start,
                'end': note_end,
                'velocity': velocity
            })

    # Sort the notes by start time
    notes.sort(key=lambda x: x['start'])
    return notes




""" Wrting a peekable Generator for midi object"""
class PeekableGenerator:
    def __init__(self, generator):
        self._generator = generator
        self._next_item = None
        self._has_next = False
        self._advance()

    def _advance(self):
        try:
            self._next_item = self._generator.__next__()
            self._has_next = True
        except StopIteration:
            self._next_item = None
            self._has_next = False

    def peek(self):
        if not self._has_next:
            raise StopIteration("No more elements to peek at.")
        return self._next_item

    def __next__(self):
        if not self._has_next:
            raise StopIteration("No more elements.")
        current = self._next_item
        self._advance()
        return current

    def has_next(self):
        return self._has_next

    def __iter__(self):
        yield self._next_item
        self._advance()

    

"""Generator to yield midi note object at the frame during the classification
    input: pm object
"""
def midi_yielding(midi):
    all_midi_obj :list = extract_midi_notes(midi)
    ##Processing
    for note in all_midi_obj:
        yield note


""" Yielding a list of midi notes information where it fits the time frame automatically.
    Implemented using overlapping frame structure for the training.
    Begin at 0, the frame jumping at the speed of jump_len, the size of the frame is frame_len
    This function will yield the frame at the given parameter.
"""
def frame_aligning_midi(t: int, jump_len:int, frame_len: int, sr:int, midi):
    midi_generator = PeekableGenerator(midi_yielding(midi))
    midi_labels = []

    jump_time_fraction: float = jump_len * (1/sr)
    frame_time_fraction: float = frame_len * (1/sr)
    framing = [t*jump_time_fraction, t*jump_time_fraction + frame_time_fraction]
    
    last_note = 0
    while midi_generator.has_next() or last_note > framing[0]:
        
        while midi_generator.has_next() and midi_generator.peek()['start'] >= framing[0] and midi_generator.peek()['start'] < framing[1]:
            try:
                midi_labels.append(midi_generator.__next__())
            except StopIteration:
                print("Generator exhausted, no Midi Objectis being added to the label")
                break
        
        ## Yielding the list of midi notes that are fitted in side the frame
        yield midi_labels

        midi_labels.sort(key=(lambda x: x['end']))
        ##calculating the next frame time step and removing the items from the previous frame
        framing [0] += jump_time_fraction
        framing [1] += jump_time_fraction
        while len(midi_labels) > 0 and midi_labels[0]['end'] < framing[0]:
            midi_labels.pop(0)
        
        if len(midi_labels) > 0 and midi_labels[0]['end'] > framing[0]:
            last_note = midi_labels[0]['end']

    
        
  



    
""" This function returns the aligned frame at the wav data,"""
def frame_aligning_wav(t: int, jump_len: int, frame_len: int, wav):
    begin = t * jump_len
    return wav[begin: begin + frame_len]


""" This function returns the amount seconds of audio data from the wav, began on t, while using  """

def audio_segment_of(t: int, wav, seconds: float, sr: int, jump_len: int = 512, frame_len: int = 2048, ):
    size = int(seconds*sr)
    begin = t*jump_len
    
    return wav[begin: begin + size]

def audio_segment_between(begin, end, wav, sr):
    return wav[int(begin*sr): int(end*sr)]

    
def extract_mel_spectrogram(audio, sr, n_mels=88, hop_length=512, n_fft=4096): ##_fft is the frame length 
    """
    Extract a mel-spectrogram from raw audio.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
    )
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def extract_cqt_spectrogram(audio, sr, n_bins=88, bins_per_octave=12, hop_length=512):
    """ 
    Extract a CQT_spectrogram from raw audio
    """
    cqt = librosa.cqt(
        y=audio, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length
    )
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return cqt_db

def extract_hcqt_spectrogram(audio, sr, n_bins=88, bins_per_octave=12, hop_length=512, harmonics=[1, 2, 3]):
    """
    Extract a Harmonic Constant-Q Transform (HCQT) spectrogram.
    """
    hcqt = []
    max_bins = n_bins
    max_frame = 0
    for h in harmonics:
        cqt = librosa.cqt(audio, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=librosa.note_to_hz('C1') * h)
        hcqt.append(cqt)
        max_frame = max(max_frame, cqt.shape[1]) 
    
    #Stack and align all of the cqt
    aligned_hcqt = []
    for cqt in hcqt:
        # Pad frequency bins
        if cqt.shape[0] < max_bins:
            cqt = np.pad(cqt, ((0, max_bins - cqt.shape[0]), (0, 0)), mode="constant")
        # Pad temporal frames
        if cqt.shape[1] < max_frame:
            cqt = np.pad(cqt, ((0, 0), (0, max_frame - cqt.shape[1])), mode="constant")
        # Crop to align dimensions (optional, if needed)
        cqt = cqt[:max_bins, :max_frame]
        aligned_hcqt.append(cqt)
    
    aligned_hcqt = np.stack(aligned_hcqt, axis=0)
    
    # Convert to dB
    hcqt_db = librosa.amplitude_to_db(np.abs(aligned_hcqt), ref=np.max)
    return hcqt_db


def extract_chroma_gram_cqt(audio, sr, hop_length=512 ):
    chromagram = librosa.feature.chroma_cqt(audio, sr=sr, hop_length=hop_length)
    return chromagram





def add_gaussian_noise(audio, noise_level=0.0006):
    
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise


