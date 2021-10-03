import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras import activations
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
import cv2
import os

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("train/",
                                            target_size=(200,200),
                                            batch_size=3,
                                            class_mode='binary')
            
validation_dataset = validation.flow_from_directory("validate/",
                                            target_size=(200,200),
                                            batch_size=3,
                                            class_mode='binary')

model = tf.keras.models.Sequential()
model.add(keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)))
model.add(keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(keras.layers.Conv2D(64,(3,3),activation='sigmoid'))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=RMSprop(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy'])


model_fit = model.fit(train_dataset,
                        steps_per_epoch=2,
                        epochs=10,
                        validation_data=validation_dataset)
path_dir = "test/"
for i in os.listdir(path_dir):
    img = image.load_img(path_dir + "/" + i)
    pic = img

    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])

    val = model.predict(images)
    print(val)
    
    plt.imshow(pic)
    if val == 1:
        plt.title("a fucking dog")
    elif val == 2:
        plt.title("a fucking human")
    else:
        plt.title("What the fuck is this?")
    plt.show()
