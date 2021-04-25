from tensorflow.keras.layers import (
    Convolution2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    ConvLSTM2D,
    Conv2D,
    LSTM,
    TimeDistributed
)
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
import simplejson as sj
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import numpy as np
from VidToFrame import VideoToFrames
from tensorflow.keras.utils import plot_model
import os

# def create_model(): #main one
#     model = Sequential()
#
#     model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='Same',
#                      activation='relu', input_shape=(28, 28, 1)))
#     model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='Same',
#                      activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
#                      activation='relu'))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
#                      activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(256, activation="relu"))
#     model.add(Dense(3, activation="softmax"))
#     return model

def create_model():  # main one
    model = Sequential()

    model.add((Convolution2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu')))
    model.add((MaxPooling2D(pool_size=(7, 7)))) # replaced from 2, 2
    model.add((Convolution2D(64, (3, 3), activation='relu')))
    model.add((MaxPooling2D(pool_size=(3, 3)))) # replaced from 2, 2
    model.add((Convolution2D(128, (3, 3), activation='relu')))
    model.add((MaxPooling2D(pool_size=(3, 3)))) # replaced from 2, 2
    model.add(tf.keras.layers.GlobalAveragePooling2D()) #replace with flatten()
    model.add(Dropout(0.5))
    model.add((BatchNormalization()))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation='softmax')) #-------------------------------------
    return model


def train(model, train_datagen, test_datagen):
    model.compile(optimizer='sgd', loss='kullback_leibler_divergence', metrics=['accuracy'])
    model.fit_generator(
        train_datagen,
        steps_per_epoch=None,
        epochs=50,
        verbose=1,
        validation_data=test_datagen,
        validation_steps=None
    )

def save_model(model):
    print("Saving...")
    model.save_weights("model.h5")
    print(" [*] Weights")
    open("model.json", "w").write(
            sj.dumps(sj.loads(model.to_json()), indent=4)
    )
    print(" [*] Model")


def load_model():
    print("Loading...")
    json_file = open("model.json", "r")
    model = model_from_json(json_file.read())
    print(" [*] Model")
    model.load_weights("model.h5")
    print(" [*] Weights")
    json_file.close()
    return model

def EvaluateModel():
# load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    weights=loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
# evaluate loaded model on test data
    loaded_model.compile(loss='kullback_leibler_divergence', optimizer='sgd', metrics=['accuracy'])
    score = loaded_model.evaluate(testingData(), weights)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

def RecognizeOnDirectory():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    weights = loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    folder_path = "C:/Users/Bilal/GaitRecognition/00_3"
    import os
    from tensorflow.keras.preprocessing import image
    images = []
    for img in os.listdir(folder_path):
        img = os.path.join(folder_path, img)
        img = image.load_img(img, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        images.append(img)

    count = 0
    images = np.vstack(images)
    classes = loaded_model.predict_classes(images)
    print(classes)
    testy = np.asarray(classes)
    print(len(classes))
    print("class 0, 1, 2 values: greater is predicted person: ")
    wow1 = np.count_nonzero(classes == 0)
    print(wow1)
    wow2 = np.count_nonzero(classes == 1)
    print(wow2)
    wow3 = np.count_nonzero(classes == 2)
    print(wow3)
    predicted_class_indices = np.bincount(classes).argmax()
    print("Class is: ")
    print(predicted_class_indices)
    if (wow1/len(classes) >= 0.75 and predicted_class_indices==0):
        (
            print("person is Hamza")
        )
    if (wow1/len(classes) <= 0.75):
        (
            print("Unidentified Person")
        )

def ActivateGPU():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    config.gpu_options.allow_growth = True
    K.set_session(sess)

# this is the augmentation configuration we will use for training
def trainingData():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        #shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)
    train_generator = train_datagen.flow_from_directory(
        './Images',  # this is the target directory
        target_size=(224, 224),
        batch_size=16,
        color_mode='rgb',
        class_mode='categorical')
    return train_generator

def testingData():
    test_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = test_datagen.flow_from_directory(
        './test',
        target_size=(224, 224),
        color_mode='rgb',
        batch_size=16,
        class_mode='categorical')
    return validation_generator

def plotmodel():
    model = create_model()
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    plot_model(model, to_file='model_plot.bmp', show_shapes=True, show_layer_names=True)

def activategpu():
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    # config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

def main():
    activategpu()
    #model = create_model()
    #trainy = trainingData()
    #testy = testingData()
    #train(model, trainy, testy)
    #save_model(model)
    #EvaluateModel()
    RecognizeOnDirectory()
    #call when input is video
    #obj = VideoToFrames()
    #obj.run()

if __name__ == "__main__":
    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())

    main()