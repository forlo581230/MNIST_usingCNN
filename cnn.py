#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alex
"""

#Loading mnist datasets
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()

x_Train4D = x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')

x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)


#Creating model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=16,
                   kernel_size=(5,5),
                   padding='same',
                   input_shape=(28,28,1),
                   activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
#Watching model detail
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=x_Train4D_normalize,
                          y=y_TrainOneHot, validation_split=0.2, epochs=10, batch_size=300, verbose=2)


#
import matplotlib.pyplot as plt
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel(train)
    plt.ylabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')

#Evaluating model accuracy
scores = model.evaluate(x_Test4D_normalize, y_TestOneHot)
scores[1]

#watching dataset image
import matplotlib.pyplot as plt
def plot_images_labels_prediction(images, labels, prediction, idx ,num=10):
    fig = plt.gcf();
    fig.set_size_inches(12, 14)
    if num>25: num=25
    for i in range(0,num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        title= "label=" + str(labels[idx])
        if len(prediction) > 0:
            title += ", predict="+str(prediction[idx])
        
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show();

#display 10 predict data starting from 0 
prediction = model.predict_classes(x_Test4D_normalize)
prediction[:10]
plot_images_labels_prediction(x_Test, y_Test,prediction,0 ,10);