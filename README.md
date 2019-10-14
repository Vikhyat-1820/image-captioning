# Image Captioning Model
This is implementation of deep learning model for image caption.The takes image as input predict the Caption for it.
# Dataset
[Flicker8k_Image_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
[Flicker8k_Text_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
## There is Step by Step process to build Image Captioner
First import all the Libraries used in the model
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
import string
from keras.utils import to_categorical
from keras.layers import Input,Dropout,Dense,LSTM,Embedding
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import os
import pickle
from keras.preprocessing.text import Tokenizer
```
#### I am using Colab for our Creating model,so I have to mount my data from Google Drive
#### https://colab.research.google.com/notebooks/welcome.ipynb#recent=true

Then after this we Define a Function to extract Text convert them into a Python Dictionary with image_name as key and list of Descriptions as Dataset

