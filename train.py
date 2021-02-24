'''
Cough Detection model - Training Pipeline

This is a pipeline to train a cough detection model - essentially an audio classification CNN - using the 
available extracted audio segments. 

Required inputs are the paths to the audio files. The pipeline extracts MFCCs features and uses them to train the model.

Created and maintained by NK 
'''
# Dependencies:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa as lb
import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
#%% Set some default parameters:

HEIGHT = 20
WIDTH = 44
INPUT_SHAPE = (HEIGHT, WIDTH, 1)

'''
IF RUNNING THIS CODE MAKE SURE TO SET THE PATHS TO THE APPROPRIATE DIRECTORIES
'''

PATH_0 = 'path/to/positives'
PATH_1 = 'path/to/negatives'
PATH_TEST = 'path/to/test/files'

SAVE_PATH = './trained_models/model_CoughDetect_2Conv.h5.h5'

#%%

def get_MFCC(path):
    '''
    Extract MFCCs from audio file in path.
    '''
    # Load audio, default sample rate is 22050 Hz
    sound, sample_rate = lb.load(path)
    
    ''' 
    Normalize within range 0,1. The *if* statement is to avoid getting 
    *nan* values from blank files that have made it to the dataset.
    '''
    if np.max(sound) != np.min(sound):
        sound /= np.max(np.abs(sound))
    
    # Calculate MFCCs
    mfcc = lb.feature.mfcc(y=sound, sr=sample_rate)
    
    return mfcc

#%%
def get_features(path):
    filepaths = []
    '''Get MFCCs from files in specified path, store them in 
    a numpy array '''
    print('Extracting features...')
    for root, dirs, files in os.walk(path):
        for name in files:
            filepaths.append(os.path.join(root,name))
            
    mfccs = []
    # Get MFCCs
    for i, filepath in enumerate(filepaths):
        mfccs.append(get_MFCC(filepath))
        if i % 10 == 0:
            print(f'Extracted features from {i} files...')
    # Store in numpy
    mfccs = np.array(mfccs)
    print('Done extracting features.')
    return mfccs

#%%
def generate_labels(mfcc_0, mfcc_1):
    print('Generating labels...')
    # Make labels for each class
    y_0 = np.zeros(mfcc_0.shape[0])
    y_1 = np.ones(mfcc_1.shape[0])
    # Join up arrays
    y = np.concatenate((y_0, y_1))
    y = pd.Series(y, dtype=int)
    y = pd.get_dummies(y).values
    
    return y

#%%
def join_classes(mfcc_0, mfcc_1):
    # Join classes
    X = np.concatenate((mfcc_0, mfcc_1))
    # Expand dimension
    X = np.expand_dims(X, axis=-1)
    
    return X

#%%
def build_model(input_shape, 
                loss="categorical_crossentropy", 
                optimizer = 'adam',
                learning_rate=0.001
                ):
    """
    Build the Neural Network using keras.
    
    args:
    input_shape (tuple): Shape of array representing a sample
    loss (str): Name of the loss function to use
    optimizer (str): Name of optimizer to use
    learning_rate (float): the learning rate
    
    returns:
    Tensorflow model
    """
    print('Building Convolutional Neural Network...')
    # Input stage
    mfcc_input=keras.layers.Input(shape=(20,44,1), name = "mfccInput")
    
    # 1st Convolution stage
    x=keras.layers.Conv2D(32,3,strides=(1,1),padding='same', name = 'conv1')(mfcc_input)
    x=keras.layers.BatchNormalization(name = 'bnorm1')(x)
    x=keras.layers.Activation(keras.activations.relu, name = 'act1')(x)
    x=keras.layers.MaxPooling2D(pool_size=2,padding='valid', name = 'pool1')(x)
    x=keras.layers.Dropout(0.1, name='drop1')(x)
    
    # 2nd Convolution stage
    x=keras.layers.Conv2D(64,3,strides=(1,1),padding='same', name = 'conv2')(x)
    x=keras.layers.BatchNormalization(name = 'bnorm2')(x)
    x=keras.layers.Activation(keras.activations.relu, name = 'act2')(x)
    x=keras.layers.MaxPooling2D(pool_size=2,padding='valid', name = 'pool2')(x)
    x=keras.layers.Dropout(0.1, name='drop2')(x)
    
    # Fully Connected stage
    x=keras.layers.Flatten(name = 'flatten')(x)
    x=keras.layers.Dense(units = 64, activation = 'relu', name = 'dense')(x)
    x=keras.layers.Dropout(0.2, name='drop4')(x)
    
    # Output stage
    mfcc_output=keras.layers.Dense(2, activation='softmax', name = 'out')(x)
    
    model=keras.Model(mfcc_input, mfcc_output, name="mfccModel")


    model=keras.Model(mfcc_input, mfcc_output, name="mfccModel")
    
    model.compile(loss=loss, 
                       optimizer=optimizer,
                       metrics= ['accuracy'])
    
    keras.backend.set_value(model.optimizer.learning_rate, learning_rate)
    
    return model

#%%
def train(model, 
          X,
          y,
          val_split = 0.2,
          epochs = 20, 
          batch_size = 24,
          patience = 5,
          ):
    
    """
    Train the model
    
    args:
    epochs (int): Number of training epochs
    batch_size (int): Samples per batch
    patience (int): Number of epochs to wait before early stopping
    training_data (object): Tensorflow data generator for training data
    validation_data (object): Tensorflow data generator for validation data
    
    returns:
    Training history
    """
    
    X_train, X_val, y_train, y_val = train_test_split(X, 
                                                      y, 
                                                      test_size=val_split)
    
    # Callback for early stopping
    model_callbacks = [keras.callbacks.EarlyStopping(patience=patience),
                       keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=0.00001,mode='min')
                       ]
    
    print('Training model...')
    # Training
    history = model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=model_callbacks,
            verbose = 1)
    
    return history
#%%
def plot_history(history):
    """
    Plot accuracy and loss over the training process
    
    args:
    history: Training history of the model
    returns:
    Plots
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
  
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
  
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    
    plt.show()
#%%
def test(model):
    
    # Load test data
    MFCC_TEST = get_features(PATH_TEST)
    X_test = np.expand_dims(MFCC_TEST, axis=-1)

    y_test = np.load('y_test.npy')
    y_test = pd.Series(y_test, dtype=int)
    y_test = pd.get_dummies(y_test).values
    
    # Evaluate
    score, acc = model.evaluate(X_test, y_test)
    print('Test score:', score)
    print('Test accuracy:', acc) 
#%%
def main():
    # Extract features:
    
    # Cough samples
    MFCC_0 = get_features(PATH_0)
    # Non cough samples
    MFCC_1 = get_features(PATH_1)
    # Join classes
    X = join_classes(MFCC_0, MFCC_1)
    
    # Generate labels
    y = generate_labels(MFCC_0, MFCC_1)
    
  
    # Build CNN:
    CNN = build_model(INPUT_SHAPE)
    
    # Train the model:
    HISTORY = train(CNN, X, y)
    
    # Plot history:
    plot_history(HISTORY)
    
    # Test model:
    #test(CNN)    
    
    # Save trained model:
    CNN.save(SAVE_PATH)
#%%
if __name__ == "__main__":
    main()
