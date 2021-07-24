import cv2
import os
import numpy as np
from glob import glob
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model

from keras.models import model_from_json
from pickle import load
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences

from keras.applications import MobileNet
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications.mobilenet import preprocess_input
from keras import optimizers  



def predict():
    base_model=MobileNet(weights='imagenet',include_top=False)

    x=base_model.output
    x=GlobalAveragePooling2D()(x)

    preds=Dense(10,activation='softmax')(x) #final layer with softmax activation
    model = Model(inputs=base_model.input, outputs=preds)

    model.summary()

    sgd = optimizers.SGD(lr = 0.005) # try 0.01 - didn't converge and 0.005 , 0.001 best acc of 11%

    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy']) # create object

    # labels is the image array
    test_image = []
    i = 0

    files = os.listdir('Test')
    for i in range(1):
        print ('Image number:',i)
        im = Image.open(r'Test/'+files[i])
        print(im.filename)
        img = cv2.imread('Test/'+files[i])
        #img = color.rgb2gray(img)
        img = img[50:,120:-50]
        img = cv2.resize(img,(128,128))
        test_image.append(img)
    test = []

    for img in test_image:
        test.append(img)

    model.load_weights('mobilenet_sgd_nolayers.hdf5')

    test = np.array(test).reshape(-1,128,128,3)
    prediction = model.predict(test)

    tags = { "C0": "safe driving",
    "C1": "texting - right",
    "C2": "talking on the phone - right",
    "C3": "texting - left",
    "C4": "talking on the phone - left",
    "C5": "operating the radio",
    "C6": "drinking",
    "C7": "reaching behind",
    "C8": "hair and makeup",
    "C9": "talking to passenger" }

    predicted_class = int(str(np.where(prediction[0] == np.amax(prediction[0]))[0][0]))

    return(predicted_class)
