import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from keras.utils import np_utils
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import os
import argparse
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
import time 


# In[2]:


def data_load(path):
    img=[]
    label=[]
    class_names = os.listdir(path)
    l=0
    for folder in class_names:
        images= glob.glob(path+folder+'/*.jpg')
        for image in images:
            S_img=cv2.imread(image)
            img.append(S_img)
            label.append(l)
        l=l+1
    return(img,label,class_names)
def data_prep(img,label,class_names):
    train_images, test_images, train_labels, test_labels = train_test_split(img, label, test_size=0.25, random_state=42)

    #Hot encode labels
    train_labels = np_utils.to_categorical(train_labels)
    test_labels = np_utils.to_categorical(test_labels)

    # Normalize pixel values between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #Just checking
    print("Train images size:", train_images.shape)
    print("Train labels size:", train_labels.shape)
    print("Test images size:", test_images.shape)
    print("Test label size:", test_labels.shape)
    return(train_images,test_images,train_labels,test_labels)
def i_data_prep(img,label,class_names):
    labels=np_utils.to_categorical(label)
    img=img/255.0
    return(img,labels)

def sample_plot(train_images, train_labels):

    #get_ipython().run_line_magic('matplotlib', 'inline')
    #Show first 25 training images below
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        x = np.array(train_labels[i]).tolist()
        index = x.index(1.)
        plt.xlabel(class_names[index])
        plt.savefig("Samples from Dataset.png")

def model_init(INET, class_names):  
    Model = tf.keras.Sequential()
    Model.add(INET)
    Model.add(layers.Flatten())
    Model.add(layers.Dense(128, activation = 'relu')) #First hidden layer
    Model.add(layers.Dense(128, activation = 'relu')) #Second hidden layer
    Model.add(layers.Dropout(0.2)) #Third hidden layer
    Model.add(layers.Dense(128, activation = 'relu')) #Fourth hidden layer
    Model.add(layers.Dense(128, activation = 'relu')) #Fifth hidden layer
    Model.add(layers.Dense(len(class_names), activation=tf.nn.softmax)) #Output layer
    return(Model)
def A_L_Plot(history):
    
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss function value on training data')
    plt.xlabel('epoch')
    # plt.savefig('Training_loss.png')
    plt.show()
    # #### Plot the accuracy (on training data) with respect to the epoch number

    # In[12]:


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(acc,label='Training Accuracy')
    plt.plot(val_acc,label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Model Training and Validation Accuracy')
    plt.xlabel('epoch')
    # plt.savefig('Traning_accuracy.png')
    plt.show()
if __name__ == '__main__':


        parser = argparse.ArgumentParser(description='MobileNetV2 Keras ') 
        parser.add_argument("--model_dir", type= str , default='./checkpoint/', help='path to the save or load the chekpoint')    
        parser.add_argument("--data", type= str, default='./data/', help='Dataset location')
        parser.add_argument("--infer-data", type= str, default='./data/', help='data for infrence ')
        parser.add_argument("--inps", type= str, default='test', help='select test, train, infer')
        parser.add_argument("--b_s", type=int,default=32, help="Batch Size")
        parser.add_argument("--e", type=int,default=1, help="Epochs")
        args = parser.parse_args()
        print(args)
        img,label,class_names = data_load(args.data)
        img=np.array(img)
        train_images,test_images,train_labels,test_labels=data_prep(img,label,class_names)
        #sample_plot(train_images,train_labels)

        if args.inps == 'tune' :
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)
                INET=tf.keras.applications.inception_v3.InceptionV3(
                                                                    include_top=False,
                                                                    weights='imagenet',
                                                                    input_tensor=None,
                                                                    input_shape=(96,96,3),
                                                                    pooling=None,
                                                                    classes=len(class_names),
                                                                    classifier_activation='softmax'
                                                                    )
                Model=model_init(INET,class_names)
                early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
                checkpoint = ModelCheckpoint(args.model_dir, monitor='loss', verbose=1, save_best_only=True, mode='min')

                Model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
            else:
                Model=tf.keras.models.load_model(args.model_dir, custom_objects=None, compile=True, options=None)
                early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
                checkpoint = ModelCheckpoint(args.model_dir, monitor='loss', verbose=1, save_best_only=True, mode='min')
            



            os.system('clear')
            print('************************ Fine tuning ************************')
            print(Model.summary())
            history = Model.fit(train_images, 
                      train_labels,
                      epochs=args.e,
                      batch_size=args.b_s,
                      verbose=1,
                      validation_data=(test_images,test_labels),
                      callbacks = [early_stop, checkpoint]
                   )
            A_L_Plot(history)
            
        if args.inps == 'test':
            os.system('clear')
            print('************************ testing ************************')

            M1=tf.keras.models.load_model(args.model_dir, custom_objects=None, compile=True, options=None)
            [a,b,c,d]=(np.shape(test_images))
            t=time.time()
            L_hat= M1.predict(test_images)
            t=time.time()-t
            v=t/a
            print('latency:', v)
            L_hat = np.argmax(L_hat, axis=1)
            print(np.unique(L_hat))
            test_labels1=np.argmax(test_labels,axis=1)
            print(np.unique(L_hat))
            print('precision:',precision_score(test_labels1,L_hat, average="macro"))
            print('Recall:',recall_score(test_labels1,L_hat, average="macro"))
            print('F1 Sore :', f1_score(test_labels1,L_hat, average="macro"))
            print('Accuracy:', accuracy_score(test_labels1,L_hat))
            print('confusion_matrix:',confusion_matrix(test_labels1,L_hat))

        if args.inps == 'train':

            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)

            INET=tf.keras.applications.inception_v3.InceptionV3(
                                                                include_top=False,
                                                                weights=None,
                                                                input_tensor=None,
                                                                input_shape=(96,96,3),
                                                                pooling=None,
                                                                classes=len(class_names),
                                                                classifier_activation='softmax'
                                                                )
            Model = model_init(INET, class_names)
            early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
            checkpoint = ModelCheckpoint(args.model_dir, monitor='loss', verbose=1, save_best_only=True, mode='min')

            Model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            os.system('clear')
            print('************************ Traning from Scratch************************')
            history = Model.fit(train_images, 
                      train_labels,
                      epochs=args.e,
                      batch_size=args.b_s,
                      verbose=1,
                      validation_data=(test_images,test_labels),
                      callbacks = [early_stop,checkpoint]
                   )
            A_L_Plot(history)

        if args.inps  == 'resume':


            if not os.path.exists(args.model_dir):
                os.system('clear')
                print('********************************************** no  check point  found ********************************************** ')
            else:
                Model=tf.keras.models.load_model(args.model_dir, custom_objects=None, compile=True, options=None)
                early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
                checkpoint = ModelCheckpoint(args.model_dir, monitor='loss', verbose=1, save_best_only=True, mode='min')
                os.system('clear')
                print('************************ Resuming Traning from check point************************')
                history = Model.fit(train_images, 
                          train_labels,
                          epochs=args.e,
                          batch_size=args.b_s,
                          verbose=1,
                          validation_data=(test_images,test_labels),
                          callbacks = [early_stop,checkpoint]
                       )
                A_L_Plot(history)

