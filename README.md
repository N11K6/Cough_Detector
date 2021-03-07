# Cough detection using a Convolutional Neural Network

My first piece of work in Eupnoos was to develop a cough monitoring system. The broader scope of this application was to perform an overtime monitoring of one's sleep through the frequency and quality of their coughing. Recordings are sent from a smartphone to the server, where coughing events are identified, isolated, and sent for further processing to determine severity and perform a diagnosis.

This project regards the initial cough detection stage of this system. An input audio stream is processed to separate events from silence. The Mel Frequency Cepstral Coefficients (MFCCs) from each of these events are extracted and used as input to a Convolutional Neural Network. The network then performs a classification task, to determine wether the sound corresponds to a coughing event or something not of interest.

## Data

Allocating and using data of medical interest is always complicated, but for this initial task, a collection containing audio of coughing is sufficient. Fortunately, for the simple task of differentiating cough sounds from anything else, there exists an adequate amount of data that can be freely used and extracted. All the audio collected here has been segmented either from recordings supplied by Eupnoos or by Youtube videos and openly available sound databases.

## Approach

For each audio file, the singular audio events are isolated by considering a minimum interval of silence, and a dB threshold. These events are then padded or cut short in order to obtain uniformity of length.

The features extracted from each file are its MFCCs, which offer a compact container of information on the content of the sound, especially if it is human activity-related, and have been a consistent choice for input when training neural networks.

The architecture of the neural network consists of two main 2D Convolutional stages, followed by a fully connected stage that connects to the 2-neuron output layer, corresponding to the two classes (positive or negative).

![alt text](https://github.com/N11K6/Cough_Detector/blob/main/images/model_schematic.png?raw=true)

The model is trained on the available data, in order to be able to differentiate between an audio event belonging to a cough, or anything else.

## Results

Training results have been giving a validation accuracy of 98% percent and F1 score at 0.93 , and thus far, testing using unknown audio has yielded an accuracy of 92% / F1 of 0.91. As more data becomes available for use, the model, and these statistics are updated.

## Contents

* *Notebooks* : Contains the *.ipbynb* notebooks that outline the process and methodology behind assembling the dataset and training the model.
* *audio_files* : Is the directory where any audio files that can be used to demonstrate the model are stored. 
* *demo_video* : Contains a short screen capture video of the detection system in deployment.
* *Deploy_w_Flask* : Contains the necessary code and the client script to deploy the cough detector on Docker with NGINX, using the Flask API. Note that any audio files used in the example are still called from the *audio_files* directory.
* *images* : Contains images and schematics for presentation.
* *trained_models* : Contains the trained models saved in *.h5* format.
* *training_mfccs* : Contains a number of extracted MFCCs that can be used to run the code and train the model.
* **utils.py** : Is a python script containing the functions used to help assemble a dataset of audio events to train the model.
* **train.py** : Is the training pipeline used to build and train the model from the available audio files.
* **train_from_mfccs.py** : Is an alternative training pipeline that can be used with the MFCCs stored in the **training_mfccs** directory. Features are directly loaded and the model is built and trained without the need to have the audio files stored.
* **trained_model_demo.py** : Is a program for a brief demonstration of the model, using an input audio file to detect coughing events.
