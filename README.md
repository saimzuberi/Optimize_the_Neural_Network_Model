# Optimize_the_Neural_Network_Model

To predict whether Alphabet Soup funding applicants will be successful, you will create a binary classification model using a deep neural network.

This challenge consists of three technical deliverables. You will do the following:

Preprocess data for a neural network model.

Use the model-fit-predict pattern to compile and evaluate a binary classification model.

Optimise the model.

## Imports 
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder

## Neural Network Set up 
Converting Object type Features into Float using OneHotEncoder and StandardScaler
Using 
        Relu on the 1st Layer
        Relu on hidden layer 
        Sigmoid on Output Layer for binary output

Complie the model for loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]

## Result 

268/268 - 1s - loss: 0.5646 - accuracy: 0.7283 - 631ms/epoch - 2ms/step
Loss: 0.5645801424980164, Accuracy: 0.7282798886299133

## Neural Network A1 
Converting Object type Features into Float using OneHotEncoder and StandardScaler
Using 
        Relu on the 1st Layer
        Neurons = (number_input_features + 1) // 2 
        Relu on hidden layer 
        Neurons = (hidden_nodes_layer1 + 1) // 2
        Sigmoid on Output Layer for binary output
        Neurons = 1

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 58)                6786      
                                                                 
 dense_1 (Dense)             (None, 29)                1711      
                                                                 
 dense_2 (Dense)             (None, 1)                 30        
                                                                 
=================================================================
Total params: 8,527
Trainable params: 8,527
Non-trainable params: 0
_________________________________________________________________

Complie the model for loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]

## Result 

268/268 - 1s - loss: 0.5646 - accuracy: 0.7283 - 631ms/epoch - 2ms/step
Loss: 0.5645801424980164, Accuracy: 0.7282798886299133

## Neural Network A2

Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_3 (Dense)             (None, 96)                11232     
                                                                 
 dropout (Dropout)           (None, 96)                0         
                                                                 
 dense_4 (Dense)             (None, 54)                5238      
                                                                 
 dropout_1 (Dropout)         (None, 54)                0         
                                                                 
 dense_5 (Dense)             (None, 1)                 55        
                                                                 
=================================================================
Total params: 16,525
Trainable params: 16,525
Non-trainable params: 0
_________________________________________________________________

## Result 

Alternative Model 1 Results
268/268 - 0s - loss: 0.7103 - accuracy: 0.5532 - 323ms/epoch - 1ms/step
Loss: 0.7102518081665039, Accuracy: 0.5532361268997192

## Neural Network A2

Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_6 (Dense)             (None, 32)                3744      
                                                                 
 dense_7 (Dense)             (None, 16)                528       
                                                                 
 dense_8 (Dense)             (None, 8)                 136       
                                                                 
 dense_9 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 4,417
Trainable params: 4,417
Non-trainable params: 0
_________________________________________________________________

## Result 

Alternative Model 2 Results
268/268 - 1s - loss: 0.7048 - accuracy: 0.4945 - 514ms/epoch - 2ms/step
Loss: 0.7048028707504272, Accuracy: 0.4944606423377991
