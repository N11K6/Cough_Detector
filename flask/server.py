#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server for a Flask API implementation of the cough detection model.

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

def identify():
    
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
