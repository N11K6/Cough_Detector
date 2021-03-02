#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client to be used for a Flask API implementation of the cough detection model.

@author: nk
"""

#%% Dependencies:
import requests

#%%
URL = "http://127.0.0.1:5000/"
PATH_TO_AUDIO = "./audio_files/audio_test_7s.wav"

#%%
def present_results(outcomes):
    '''
    Prints out the results in a tidy format.
    args:
        outcomes (list) : list of predictions (0 or 1) corresponding to either
                         a cough event or a non-cough related event.
    '''
    # Count number of positives detected:
    total = len(outcomes)
    positives = 0
    for i in range(total):
        if outcomes[i] == 0:
            positives += 1
    if positives >= 1:
        print('Detected ', positives,' cough events out of ',
              total, ' total audio events.')
    else:
        print('No cough events detected in ', total, ' total audio events.')
        
#%%
if __name__ == "__main__":

    # open files
    file = open(PATH_TO_AUDIO, "rb")

    # package stuff to send and perform POST request
    values = {"file": (PATH_TO_AUDIO, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()
    
    outcomes = data["outcomes"]
    
    present_results(outcomes)

