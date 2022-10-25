#Intro

# IMG_SIZE is determined by EfficientNet model choice
IMG_SIZE = 224

import tensorflow as tf
import numpy as np

strategy = tf.distribute.MirroredStrategy()


#%%
#Set prefix
prefix = "Binary"
#%%
#Set different params

batch_size = 128
learning_rate = 0.002
epochs = 80
data_directory = prefix
train_from_scratch = True
name = prefix + ("from_scratch" if train_from_scratch else "pretrained")
print(name)
##

#%%
#Load dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0


img_size = (IMG_SIZE, IMG_SIZE)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',

    
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=img_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_directory,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',


    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=img_size,
    batch_size=batch_size,
)

import matplotlib.pyplot as plt

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

print(class_names)
print(NUM_CLASSES)

        
    
# One-hot / categorical encoding
def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label



#normalization_layer = tf.keras.layers.Rescaling(1./255)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
train_ds = train_ds.with_options(options)


train_ds = train_ds.map(lambda image, label: (tf.image.resize(image, img_size), label))
val_ds = val_ds.map(lambda image, label: (tf.image.resize(image, img_size), label))

ds_train = train_ds.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = val_ds.map(input_preprocess)


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        
plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

#%%
#Create model

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

def build_model(num_classes, train_whole, learning_rate):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = tf.keras.applications.MobileNetV3Small(include_top=False, input_tensor=inputs, weights="imagenet", minimalistic=True)

    # Freeze the pretrained weights
    model.trainable = train_whole

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.3
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="MobileNetV3Small")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


with strategy.scope():
    model = build_model(NUM_CLASSES, train_from_scratch, learning_rate)
    
model.summary()

#%%
#Fit model

hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


plot_hist(hist)


model.save("mobileNet_" + name + ".h5")


#%%
#Load model

from tensorflow.keras.models import load_model
test_model = load_model("mobileNet_" + name + ".h5")

#%%
#Test model on a single image

from PIL import Image
import timeit
import time
import numpy as np

zer = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))

sample = Image.open("sample_raw.png")
sample = np.array(sample)

for i in range(3):
    zer[0, :, :, i] = sample[:, :]
sample = zer

device_gpu_name = "/gpu:0"

device_cpu_name = "/cpu:0"



#GPU

with tf.device(device_gpu_name):
    start = time.time_ns()

    prediction = str(np.argmax(test_model(sample)))
    
    end = time.time_ns()

    perf = end - start
    print(prediction)
    print(str(perf / 1000000.0) + " ms")

#CPU
with tf.device(device_cpu_name):
    start = time.time_ns()

    prediction = str(np.argmax(test_model(sample)))
    
    end = time.time_ns()

    perf = end - start
    print(prediction)
    print(str(perf / 1000000.0) + " ms")


#%%
#Test model on multiple images

from tensorflow.keras.models import load_model


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        
        img = np.array(images[i])
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        prediction = str(np.argmax(test_model(img)))
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title("truth=" + class_names[labels[i]] + " ; pred=" + prediction)
        plt.axis("off")


plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        
        img = np.array(images[i])
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        prediction = str(np.argmax(test_model(img)))
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title("truth=" + class_names[labels[i]] + " ; pred=" + prediction)
        plt.axis("off")