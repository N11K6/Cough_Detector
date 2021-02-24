'''
Cough Detection model - Training Pipeline v2

TRAIN USING STORED MFCCs IN NUMPY ARRAYS

This pipeline can be used to build and train the cough detection model using the already extracted
MFCC features (.npy files in the training_mfccs directory).

The program loads the MFCCs, joins them to form an input tensor, compiles the model and trains it.
The trained model is then saved is .h5 format.

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
Paths to the stored MFCCs and destination path to save the trained model:
'''

PATH_0 = './training_mfccs/mfccs_pos.npy'
PATH_1 = './training_mfccs/mfccs_neg.npy'

SAVE_PATH = './trained_models/model_CoughDetect_2Conv.h5'

#%%
def load_MFCCs(path_0, path_0):
  '''
  Loads stored MFCCs in numpy arrays:
  '''
  # positives
  mfcc_0 = np.load(path_0)
  # negatives
  mfcc_1 = np.load(path_1)

  return mfcc_0, mfcc_1

#%%
def generate_labels(mfcc_0, mfcc_1):
  '''
  Generates labels for the training data 
  0 for positive
  1 for negative
  '''
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
  '''
  Joins the two classes to form an input tensor:
  '''
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
    
    # Training
    history = model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=model_callbacks,
            verbose = 2)
    
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
def main():
   
    # Load input:
    MFCC_0, MFCC_1 = load_MFCCs(PATH_0, PATH_1)

    # Join positives and negatives:
    X = join_classes(MFCC_0, MFCC_1)
    
    # Generate labels:
    y = generate_labels(MFCC_0, MFCC_1)
    
    # Build CNN:
    CNN = build_model(INPUT_SHAPE)
    
    # Train the model:
    HISTORY = train(CNN, X, y)
    
    # Plot history:
    plot_history(HISTORY)
    
    # Save trained model:
    CNN.save(SAVE_PATH)

#%%
if __name__ == "__main__":
    main()
