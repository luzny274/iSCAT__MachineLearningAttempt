import os


num_classes = 3
num_classes = 2

#%%
DATA_DIR = "SegmentedMTs_mid_Sorted_8_1"
model_folder = "models/unet_checkpoint"
#%%
DATA_DIR = "SegmentedMTs_mid_Sorted_8_1_128"
model_folder = "models/unet_checkpoint_128"
#%%

input_dir = DATA_DIR + "/Images/"
target_dir = DATA_DIR + "/Category_ids/"
img_size = (128, 128)
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print("Number of samples:", len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)
import tensorflow as tf

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

# Display input image #7
#display(Image(filename=input_img_paths[9]))

# Display auto-contrast version of corresponding target (per-pixel categories)
#img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
#display(img)

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class MyDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="rgb")
            #img = tf.image.grayscale_to_rgb(img) ##
            
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        #y = y / 127
        y[y <= 200] = 0
        y[y > 0] = 1
    
        return x, y


    
from tensorflow.keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()


import random

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = MyDataset(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = MyDataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)

#if False:
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

elem = next(iter(train_gen))

for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = elem[0][i]
    img = img / 255.0
    plt.imshow(img)
    plt.axis("off")
    
    
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = elem[1][i]
    plt.imshow(img)
    plt.axis("off")
    
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

# Train the model, doing validation at the end of each epoch.
epochs = 15

#%%
#Training


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = model_folder,
    save_best_only=True)

model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=model_checkpoint_callback)


# Generate predictions for all images in the validation set

val_gen = MyDataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)


# Display results for validation image #10
i = 10

# Display input image
display(Image(filename=val_input_img_paths[i]))

# Display ground-truth target mask
img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.


#%%
#Load model
#model_folder = "models/unet_checkpoint"
#model_folder = "models/backup/low_contrast_data/unet_checkpoint"
#model_folder = "models/deeplab_checkpoint_11_7_99.53%"

from tensorflow.keras.models import load_model
test_model = load_model(model_folder)

#%%
#Test
def infer(model, image_tensor):
    predictions = model.predict(np.expand_dims((image_tensor), axis=0))
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions

offset = 10


elem = next(iter(train_gen))

fig = plt.figure(figsize=(10, 10))
fig.suptitle('Input', fontsize=16)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = elem[0][i]
    img = img / 255.0
    plt.imshow(img)
    plt.axis("off")
    
    
fig = plt.figure(figsize=(10, 10))
fig.suptitle('Ground truth', fontsize=16)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = elem[1][i]
    plt.imshow(img)
    plt.axis("off")

    
fig = plt.figure(figsize=(10, 10))
fig.suptitle('Predicted', fontsize=16)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = elem[0][i]
    img = infer(image_tensor=img, model=test_model)
    plt.imshow(img)
    plt.axis("off")
    
    
#%%
#Test 2

from glob import glob

NUM_TRAIN_IMAGES = 46000
NUM_VAL_IMAGES = 2000
IMAGE_SIZE = 224

DATA_DIR = "SegmentedStrict_Sorted"

train_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "Images/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "Category_ids/*")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]


def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_png(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = image / 127
    else:
        image = tf.image.decode_png(image, channels=1)
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1.0
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = tf.image.grayscale_to_rgb(image)
        
    return image


offset = 10

fig = plt.figure(figsize=(10, 10))
fig.suptitle('Input', fontsize=16)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = read_image(val_images[i + offset], False).numpy() * 0.5 + 0.5
    plt.imshow(img)
    plt.axis("off")
    
fig = plt.figure(figsize=(10, 10))
fig.suptitle('Predicted', fontsize=16)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = (read_image(val_images[i + offset], False).numpy() + 1.0) * 127.5
    img = infer(image_tensor=img, model=test_model)
    plt.imshow(img)
    plt.axis("off")
    
fig = plt.figure(figsize=(10, 10))
fig.suptitle('Ground truth', fontsize=16)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    img = read_image(val_masks[i + offset], True).numpy()
    plt.imshow(img)
    plt.axis("off")


