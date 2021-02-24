#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a demo program for the cough detector model.

Inputs: PATH_TO_AUDIO : path to the audio file for examining
        PATH_TO_MODEL : path to the trained model
        
The program segments the input file to separate audio events, extracts features from each,
and makes a prediction on each one to determine wether it constitutes a coughing event.

The waveforms of the input, and the positively identified audio are displayed.

Created and maintained by NK
"""

# Dependencies:
import time
import numpy as np
import librosa as lb
import librosa.display as lbd
from tensorflow import keras
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence

#%%
'''
Necessary Inputs:

'''
PATH_TO_AUDIO = "path/to/audio.wav"
PATH_TO_MODEL = './trained_models/model_CoughDetect_2Conv_210220.h5'

#%%
def isolate_events(loadpath, min_silence_len = 500, silence_thresh = -35):
    '''
    Function to isolate singular audio events from an input file.
   
    args:
    loadpath : path to audio file
    min_silence_len : minimum time in ms to consider as silence
    silence_thresh : minimum level in dB for detecting a sound
    
    returns:
    events : list of arrays representing each audio event
    '''
    
    print('Processing audio...')
    
    time.sleep(1)
    # RAW AUDIO SHOULD BE MONO WAV AT 22050 Hz !!!!!!!!!!!!!!!!
    raw_audio = AudioSegment.from_wav(loadpath)
    
    # Present raw audio:
    audio_array, sample_rate = lb.load(loadpath, sr = None)
    plt.figure(figsize = (12, 4))
    lbd.waveplot(audio_array, sr = sample_rate)
    plt.title('Recorded Audio')
    plt.show()
    # Isolate events:
    events = split_on_silence(raw_audio, 
                                    # must be silent for at least half a second
                                    min_silence_len=min_silence_len,
                                    # consider it silent if quieter than 
                                    silence_thresh=silence_thresh
                                   )
       
    return events

#%%
def get_mfcc_features(events):
    '''
    Extracts MFCCs from audio events and returns them in a numpy array.
    
    args:
    events : list of audio event arrays
    
    returns:
    mfccs : numpy array containing extracted MFCCs
    '''
    mfccs = []
    
    for event in events:
        # Convert pydub audio array to numpy
        event = np.array(event.get_array_of_samples()).astype(np.float32)
        # Fix length to 1s
        if len(event) < 22050:
            event = np.append(event, np.zeros(22050-len(event)))
        else:
            event = event[:22050]
        # Get MFCCs
        mfcc = lb.feature.mfcc(event)
        mfccs.append(mfcc)
    
    mfccs = np.array(mfccs)
    
    return mfccs
  
#%%
def make_prediction(path_to_model, mfccs):
    '''
    Uses trained model to make predictions on the extracted features.
    
    args:
    path_to_model : path to a trained model .h5
    mfccs : numpy array containing the MFCC features to be used as input
    
    returns:
    outcomes : list of predictions for each event - positive (0) or negative (1)
    '''
    # Load model
    model = keras.models.load_model(path_to_model)
    # Expand feature dimensions
    features = np.expand_dims(mfccs, axis=-1)
    # Make predictions
    predictions = np.round(model.predict(features),0)
    
    #predictions = np.round(predictions,0)
    
    outcomes = []
    
    for prediction in predictions:
        if prediction[0] > prediction[1]:
            outcomes.append(0)
        else:
            outcomes.append(1)
    print(predictions)
    
    return outcomes
  
#%%
def present_results(events, outcomes):
    '''
    Function to present results to the user.
    
    args:
    events : list of audio event arrays
    outcomes : list of predictions, either positive (0) or negative (1) associated with each event
    '''
    # Count number of positives detected:
    positives = 0
    for i in range(len(events)):
        if outcomes[i] == 0:
            positives += 1
    
    if positives == 1: # if single positive event

        # Plot the waveforms of the positive samples:
        fig, ax = plt.subplots(nrows=positives, sharex=False, sharey=True, figsize = (8,6))
        fig.tight_layout(h_pad=8)
        
        for i in range(positives):
    
            event = np.array(events[i].get_array_of_samples()).astype(np.float32)
            lbd.waveplot(event, sr=22050)
            ax.set_title(f'Cough event #{i+1}', fontdict={'fontsize': 16, 'fontweight': 'medium'})
    
        print('Detected ', positives,' cough events out of ', len(events), ' total audio events.')
        print()
        
    elif positives > 1: # if multiple positives

        # Plot the waveforms of the positive samples:
        fig, ax = plt.subplots(nrows=positives, sharex=False, sharey=True, figsize = (8,6))
        fig.tight_layout(h_pad=8)
    
    
        for i in range(positives):
    
            event = np.array(events[i].get_array_of_samples()).astype(np.float32)
            lbd.waveplot(event, sr=22050, ax=ax[i])
            ax[i].set_title(f'Cough event #{i+1}', fontdict={'fontsize': 16, 'fontweight': 'medium'})
            #ax[i].set(title=f'Cough event #{i+1}')
    
        print('Detected ', positives,' cough events out of ', len(events), ' total audio events.')
        print()
        
    else: # if no positives found
        print('No cough events detected in ', len(events), ' total audio events.')
        
#%%
def main():
    
    # Isolate audio events from input audio:
    EVENTS = isolate_events(PATH_TO_AUDIO)
    
    print(f'Identified {len(EVENTS)} audio events.')
    print()
    time.sleep(0.5)
    
    # Extract features from each event:
    MFCCS = get_mfcc_features(EVENTS)
    print('Extracting features...')
    print()
    time.sleep(0.5)
    
    # Make predictions for each event:
    OUTCOMES = make_prediction(PATH_TO_MODEL, MFCCS)
    print('Analyzing audio events...')
    print()
    time.sleep(1.2)
    
    # Present findings:
    present_results(EVENTS, OUTCOMES)    
#%%
if __name__ == "__main__":
    main()
