#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class-based instantiation of the cough detector. Can be used for deployment on
new audio samples.

@author: nk
"""

#%% Dependencies:
import numpy as np
import librosa as lb
from tensorflow import keras
from pydub import AudioSegment
from pydub.silence import split_on_silence

#%%
PATH_TO_MODEL = "model_CoughDetect_2Conv_210220.h5"
PATH_TO_AUDIO = "audio_test_7s.wav"
#%%

class _CoughDetector:
    '''
    Singleton class for detection of coughing events in audio files using a
    trained model.
    
    param: 
        model : Trained Tensorflow/Keras model (".h5" format)
    '''
    model = None
    
    _instance = None
    
    SAMPLE_RATE = 22050
    
    def preprocess(self, path_to_audio):
        '''
        Preprocess audio file to extract MFCC features.
        args:
            path_to_audio (str): path to audio file to read.
        returns:
            mfccs (np.ndarray) : extracted features for each audio event in
                                the audio file.
        '''
        # 1. Load input audio file at set sample rate:
        raw_audio = AudioSegment.from_file(path_to_audio,
                                           format = "wav",
                                           frame_rate = self.SAMPLE_RATE)
        # 2. Split audio events on silence:
        events = split_on_silence(raw_audio, 
                                    # must be silent for at least half a second
                                    min_silence_len=500,
                                    # consider it silent if quieter than 
                                    silence_thresh=-35
                                   )
        
        # 3. Calculate MFCCs for each event:
        mfccs = []
    
        for event in events:
            # Convert pydub audio array to numpy
            event = np.array(event.get_array_of_samples()).astype(np.float32)
            # Fix length to 1s
            if len(event) < self.SAMPLE_RATE:
                event = np.append(event, np.zeros(self.SAMPLE_RATE-len(event)))
            else:
                event = event[:self.SAMPLE_RATE]
            # Get MFCCs
            mfcc = lb.feature.mfcc(event)
            mfccs.append(mfcc)
    
        mfccs = np.array(mfccs)
        # Return events (so they might be used later) and MFCCs
        return events, mfccs
    
    def identify(self, path_to_audio):
        '''
        Call trained model to classify audio events as either cough or not, 
        based on the extracted MFCC features.
        
        args:
            path_to_audio (str): path to audio file to read.
        returns: 
            outcomes (list): list of outcomes (0 for pos, 1 for neg) for each
                            identified event in the audio file.
        '''
        # Extract MFCCs from input audio:
        _, mfccs = self.preprocess(path_to_audio)
        # Expand dimensions to use as input to the model:
        mfccs = np.expand_dims(mfccs, axis=-1)
        # Predict classes with model:
        predictions = self.model.predict(mfccs)

        # Create outcomes list depending on predictions:
        outcomes = []
        for prediction in predictions:
            if prediction[0] > prediction[1]:
                outcomes.append(0) 
            else:
                outcomes.append(1)
                
        return outcomes
                    
    def present_results(self, outcomes):
        # Count number of positives detected:
        total = len(outcomes)
        positives = 0
        for i in range(total):
            if outcomes[i] == 0:
                positives += 1
                
        if positives >= 1:
            print('Detected ', positives,' cough events out of ', total, ' total audio events.')

        else:
            print('No cough events detected in ', total, ' total audio events.')
            
#%%
def CoughDetector():
    '''
    Factory function for the Cough Detector.
    -Ensures single instance of the class is used.
    -Loads trained model.
    '''
    if _CoughDetector._instance is None:
        _CoughDetector._instance = _CoughDetector()
        _CoughDetector.model = keras.models.load_model(PATH_TO_MODEL, compile = False)
        
        _CoughDetector.model.compile(loss='binary_crossentropy',
            optimizer='Adam', metrics=['accuracy'])
        
    return _CoughDetector._instance
#%%
if __name__ == "__main__":
    # create 2 instances
    CD = CoughDetector()
    CD2 = CoughDetector()
    # check that both refer to the same object
    assert CD is CD2
    
    # Identify sounds in audio file:
    OUTCOMES = CD.identify(PATH_TO_AUDIO)
    CD.present_results(OUTCOMES)