# Image Captioning Model
This is implementation of deep learning model for image caption.The takes image as input predict the Caption for it.
# Dataset
[Flicker8k_Image_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)

[Flicker8k_Text_Dataset](https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
## There is Step by Step process to build Image Captioner
Step1 -> First import all the Libraries used in the model
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

Step2 -> Then after this we Define a Function to extract Text convert them into a Python Dictionary with image_name as key and list of Descriptions as Value
we remove all the Punctuations from our Text
```
def extract_text(filename):
  description=dict()
  doc=open(filename,'r')
  text=doc.read()
  table = str.maketrans('', '', string.punctuation)
  for line in text.split('\n'):
    word=line.split()
    if len(word)<2:
      continue
    idx=word[0]
    idx=idx.split('.')[0]
    data=word[1:]
    data = [w.translate(table) for w in data]
    data=[d for d in data if len(d)>1]
    data = [d for d in data if d.isalpha()]
    desc='startseq ' +  ' '.join(data) + ' endseq'
    if idx not in description:
      description[idx]=list()
    description[idx].append(desc)
  return description
```

Then we call this function to extract text and create our Dataset
```
description=extract_text('/content/drive/My Drive/Flickr8k.token.txt')
```
Step3 -> Extract Features From Image Dataset I am using VGG16 model to extract features from image
we remove last layer from the our model to extract features from it 
```
vggmodel=VGG16()
vggmodel.layers.pop()
vggmodel=Model(inputs=vggmodel.inputs,outputs=vggmodel.layers[-1].output)
print(vggmodel.summary())
```
function to extract features
```
def feature_extractor(filename,vggmodel):
  img=image.load_img(filename,target_size=(224,224))
  img=image.img_to_array(img)
  img=img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
  img=preprocess_input(img)
  features=vggmodel.predict(img)
  return features

def image_feature(filename,vggmodel):
  features=dict()
  for img in os.listdir(filename):
    imgname=filename + '/' + img
    img=img.split('.')[0]

    features[img]=feature_extractor(imgname,vggmodel)
  return features
```

The size of the feature (output of the VGG model is 1 X 4096)

```
image_features=image_feature('/content/drive/My Drive/Flicker8k_Dataset',vggmodel)
```

since image features is large file so we save into pickel document
```
pickle.dump(image_features,open('image_features_VGG.pkl','wb'))
```

Then define function to get inbuilt dataset 


