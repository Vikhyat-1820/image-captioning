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

Then define function to get split our training and testing dataset
```
def trainimage_and_description(description,image_features,trainfilename,maxcount):
  count=0
  trainimage={}
  traindescription={}
  text=open(trainfilename,'r').read()
  text=text.split()
  for k in text:
    k=k.split('.')[0]
    if count>maxcount:
      break
    if k in image_features.keys():
      count+=1
      trainimage[k]=image_features[k]
      traindescription[k]=description[k]
  return trainimage,traindescription

def valimage_and_description(description,image_features,valfilename,maxcount):
  valimage={}
  valdescription={}
  count=0
  text=open(valfilename,'r').read()
  text=text.split()
  for k in text:
    if count>maxcount:
      break
    k=k.split('.')[0]
    if k in image_features.keys():
      count+=1
      valimage[k]=image_features[k]
      valdescription[k]=description[k]
  return valimage,valdescription
  
 ```
 
 Then we will split dataset into training and testing dataset
 1200 images and description in training dataset and 300 in validation dataset
 ```
 valimage,valdescription=valimage_and_description(description,image_features,'/content/drive/My Drive/Flickr_8k.testImages.txt',300)
trainimage,traindescription=trainimage_and_description(description,image_features,'/content/drive/My Drive/Flickr_8k.trainImages.txt',1200)
```

Then we create tokenizer to convert our text dataset into numeric form
```
def to_lines(description):
  lines=list()
  for key in description.keys():
    [lines.append(d) for d in description[key]]
  return lines

def create_tokenizer(description):
  lines=to_lines(description)
  tokenizer=Tokenizer()
  tokenizer.fit_on_texts(lines)
  return tokenizer

def maxlength(description):
  lines=to_lines(description)
  return max(len(d.split()) for d in lines)
```

Then Finally we create our Dataset which will be input to our model
```
max_len=maxlength(description)
tokenizer=create_tokenizer(description)
vocab_size=len(tokenizer.word_index)+1
def preprocess_dataset(description,image_features,max_len,tokenizer,vocab_size):
  count=0
  X1,X2,Y=list(),list(),list()
  for key,desc in description.items():
    for desc in description[key]:
      seq = tokenizer.texts_to_sequences([desc])[0]
      for i in range(1,len(seq)):
        inp=seq[:i]
        op=seq[i]
        inp=pad_sequences([inp],maxlen=max_len)[0]
        op=to_categorical([op],num_classes=vocab_size)[0]
        if key in image_features.keys():
          X1.append(image_features[key][0])
          X2.append(inp)
          Y.append(op)
    count+=1
  return np.array(X1),np.array(X2),np.array(Y)
```
Basically we create out dataset using this concept

![alt text](https://media.mnn.com/assets/images/2017/07/dog_playing_frisbee_beach.jpg.653x0_q80_crop-smart.jpg)

Caption :- A dog is playing on the beach


with these on input image and description we create this

| Input                                                              |  output       |
| ------------------------------------------------------------------ | ------------- |
| Image + startseq                                                   | A             |
| Image + startseq + A                                               | dog           |
| Image + startseq + A + dog                                         | is            |
| Image + startseq + A + dog + is                                    | playing       |
| Image + startseq + A + dog + is + playing                          | on            |
| Image + startseq + A + dog + is + playing + on                     | the           |
| Image + startseq + A + dog + is + playing + on + the               | beach         |
| Image + startseq + A + dog + is + playing + on + the + beach       | endseq        |


```
traininp1,traininp2,trainoup=preprocess_dataset(traindescription,trainimage,max_len,tokenizer,vocab_size)
```
```
valinp1,valinp2,valoup=preprocess_dataset(valdescription,valimage,max_len,tokenizer,vocab_size)
```
Then we create image caption generator model
```
def define_model(max_length,vocab_size):
	inp1 = Input(shape=(4096,))
	im1 = Dropout(0.5)(inp1)
	im2 = Dense(256, activation='relu')(im1)
	inp2 = Input(shape=(max_length,))
	tx1 = Embedding(vocab_size, 256, mask_zero=True)(inp2)
	tx2 = Dropout(0.5)(tx1)
	tx3 = LSTM(256)(tx2)
	comb1 = add([im2, tx3])
	comb2 = Dense(256, activation='relu')(comb1)
	oups = Dense(vocab_size, activation='softmax')(comb2)
	model = Model(inputs=[inp1, inp2], outputs=oups)
	model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
	print(model.summary())
	return model
```

Now lets fit our model on training dataset
```
model = define_model(max_len,vocab_size)
model.fit([traininp1, traininp2], trainoup, epochs=40,validation_data=([valinp1,valinp2],valoup))
```

Then we create a function which gives most word for corresponding int input from given tokenizer
```
def int_to_word(integer,tokenizer):
  for word, index in tokenizer.word_index.items():
    #print(index)
    if index == integer:
      return word
  return None
```

Then a function which will generate caption for given image
```
def generate_desc(photo,tokenizer,max_length,vocab_size):
  text='startseq'
  for i in range(max_length):
    seq=tokenizer.texts_to_sequences([text])[0]
    seq=pad_sequences([seq],maxlen=max_length)
    yhat=model.predict([photo,seq])
    yhat=np.argmax(yhat)
    #print(yhat)
    word=int_to_word(yhat,tokenizer)
    text=text + ' ' + word
    if word is None:
      break
    if word=='endseq':
      break
    
  return text
```




