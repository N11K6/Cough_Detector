'''
A script containing helper functions used to assemble a training dataset for the cough detection model.
Used to separate single events from large audio files, store them as .wav files of uniform length, 
extract MFCC features and store those as numpy arrays etc.
'''
# Dependencies
import numpy as np
import os
from os import walk
import librosa as lb
import librosa as lbd
import soundfile
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence
#%%
def cut_audio_chunks(loadpath, savepath, min_silence_len, silence_thresh, audioformat='mp3'):
    '''Function to split raw audio into chunks corresponding to isolated events
    takes a specified loading path, a saving path, the minimum silence time length 
    in ms, and the threshold for silence in dB.'''
    
    if audioformat == 'mp3':
        sound_file = AudioSegment.from_mp3(loadpath)
    elif audioformat == 'wav':
        sound_file = AudioSegment.from_wav(loadpath)
    
    # Make sure the directories exist to store the segmented audio:
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # split audio
    audio_chunks = split_on_silence(sound_file, 
                                    # must be silent for at least half a second
                                    min_silence_len=min_silence_len,
                                    # consider it silent if quieter than 
                                    silence_thresh=silence_thresh
                                   )
    # store the audio chunks
    for i, chunk in enumerate(audio_chunks):
        out_file = savepath+i+".wav"
        print("exporting ", out_file)
        chunk.export(out_file, format="wav")
        
#%%
def export_wavs(dirpath_from, dirpath_to, name, dur=1, sample_rate=44100):
    '''Reads audio files from a directory, segments them to a specific 
    duration and using a uniform sample rate and stores them.'''
    
    # Make directory if it doesn't exist
    if not os.path.exists(dirpath_to):
        os.makedirs(dirpath_to)
    
    filepaths = []
    # Locate existing audio
    for (_,_,filenames) in walk(dirpath_from):
        filepaths.extend(filenames)
        break
    for i, filepath in enumerate(filepaths):
        # Read in audio
        data, sr = lb.load(dirpath_from + '/'+ filepath, sr = sample_rate)
        # Set length
        if len(data) >= dur * sr:
            data = data[:sr]
        else:
            data = np.append(data, np.zeros(sr-len(data)))
        # Store .wav file
        savename = dirpath_to + '/' + name + str(i) + '.wav'
        soundfile.write(savename, data, sr)

#%%
def export_npys(dirpath_from, dirpath_to, name, dur=1, sample_rate=44100):
    '''Reads audio files from a directory, segments them to a specific 
    duration and using a uniform sample rate and stores them as numpy arrays.'''
    
    # Make directory if it doesn't exist
    if not os.path.exists(dirpath_to):
        os.makedirs(dirpath_to)
    
    filepaths = []
    # Locate wav files
    for (_,_,filenames) in walk(dirpath_from):
        filepaths.extend(filenames)
        break
    for i, filepath in enumerate(filepaths):
        # Read audio samples
        data, sr = lb.load(dirpath_from + '/'+ filepath, sr = sample_rate)
        # Set length
        if len(data) >= dur * sr:
            data = data[:sr]
        else:
            data = np.append(data, np.zeros(sr-len(data)))
        
        savename = dirpath_to + '/' + name + str(i) + '.npy'
        np.save(savename, data)
        
#%%
def demonstrate_mfcc(filepath, duration = 1, sample_rate = 44100):
    '''Calculates MFCC for a segment of specified duration from an audio file
        , using a specified sample rate, and outputs the corresponding plot.'''
    # Load audio into numpy array
    data, sr = lb.load(filepath, sr = sample_rate)
    
    # Segment audio to specified duration, zero pad if too short
    if len(data) <= sr:
        data = np.append(data, np.zeros(sr-len(data)))
    else:
        data = data[:sr]
        
    # Set up plotting environment
    fig = plt.figure(figsize = [4,4])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    #Calculate and plot MFCCs
    S = lb.feature.mfcc(y=data, sr=sr)
    lbd.specshow(lb.power_to_db(S, ref=np.max))
