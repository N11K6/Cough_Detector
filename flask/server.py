#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 18:36:37 2021

@author: nk
"""

#%% Dependencies
from flask import Flask, request, jsonify
from coughdetector import CoughDetector
import os
import random

#%%
app = Flask(__name__)

@app.route("/", methods = ["POST"])

def denoise():
    
    # get audio file and save it:
    audio_file = request.files["file"] # request file
    file_name = str(random.randint(0,100000)) # assign a random file name
    audio_file.save(file_name) # store the audio file under file name
    
    # invoke denoising AE
    CD = CoughDetector()
    
    # identify events
    events = CD.identify(file_name)
    
    # remove input audio file
    os.remove(file_name)

    # send denoised audio file data
    data = {"outcomes" : events}
    return jsonify(data)
    
#%%
if __name__ == "__main__":
    app.run(debug = False)