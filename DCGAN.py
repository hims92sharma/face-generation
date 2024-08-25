import keras
import cv2
import pandas as pd
import seaborn as sns
import os


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SeparableConv2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, add, concatenate
from tensorflow.keras.layers import LeakyReLU, Activation, Reshape
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

from GAN import GAN
from GANMonitor import GANMonitor

attribute = pd.read_csv('/Users/himanshusharma/Documents/celeba/dataset1/list_attr_celeba.csv')
bboxes = pd.read_csv('/Users/himanshusharma/Documents/celeba/dataset1/list_bbox_celeba.csv')
partitions = pd.read_csv('/Users/himanshusharma/Documents/celeba/dataset1/list_eval_partition.csv')
landmarks = pd.read_csv('/Users/himanshusharma/Documents/celeba/dataset1/list_landmarks_align_celeba.csv')
base_directory = '/Users/himanshusharma/Documents/celeba/dataset1/img_align_celeba'

# TO PLOT some images
# import glob
# import matplotlib.pyplot as plt
# import  matplotlib.image as mpimg
#
# # images = []
# # for img_path in glob.glob('/Users/himanshusharma/Documents/celeba/dataset/img_align_celeba/00000*.jpg'):
# #     images.append(mpimg.imread(img_path))
# # images = images[:9]
# # plt.figure(figsize=(20, 10))
# # columns = 5
# # for i, image in enumerate(images):
# #     plt.subplot(len(images) // columns + 1, columns, i + 1)
# #     plt.axis('off')
# #     plt.imshow(image)
# # plt.show()

print(landmarks.head())
print(partitions['partition'].value_counts())

# to PLOT image landmarks
paths_to_images ='/Users/himanshusharma/Documents/celeba/dataset/img_align_celeba/000002.jpg'
# eye_x, eye_y, eye_w, eye_h = np.array(landmarks.iloc[:, 1:5])[0]
# nose_x,	nose_y,	leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y = np.array(landmarks.iloc[:, 5:])[0]
#
# left_eye = (eye_x, eye_y)
# right_eye = (eye_w, eye_h)
# nose = (nose_x + 10,nose_y)
# left_mounth = (leftmouth_x, leftmouth_y)
# right_mounth = (rightmouth_x, rightmouth_y)
#
# example_image = cv2.imread(paths_to_images)
# original_image = example_image.copy()
#
# example_image = cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB)
#
# example_image = cv2.line(example_image, left_eye, right_eye, (0, 255, 255),1)
# example_image = cv2.line(example_image, left_eye, nose, (0, 255, 255), 1)
# example_image = cv2.line(example_image, right_eye, nose, (0, 255, 255), 1)
# example_image = cv2.line(example_image, nose, left_mounth,(0, 255, 255), 1)
# example_image = cv2.line(example_image, nose, right_mounth, (0, 255, 255), 1)
#
# plt.figure(figsize = (10, 20))
# plt.subplot(1,2,1)
# plt.axis('off')
# plt.title('original image')
# plt.imshow(original_image)
# plt.show()
# plt.subplot(1,2,2)
# plt.axis('off')
# plt.title('Image with landmarks')
# plt.imshow(example_image)
# plt.show()


# To plot boundary box in images
# example_image = cv2.imread(paths_to_images)
# original_image = example_image.copy()
# current_bbox = bboxes.query('image_id == "{}"'.format(paths_to_images.split('\\')[-1]))
# print(current_bbox)
# x, y, w, h = np.array(current_bbox.iloc[:, 1:])[0]
#
# example_image = cv2.rectangle(example_image, (x - w, y ), (w , h ), (0, 255, 255), 1)
# plt.figure(figsize = (10, 20))
# plt.subplot(1,2,1)
# plt.axis('off')
# plt.title('original image')
# plt.imshow(original_image)
# plt.show()
# plt.subplot(1,2,2)
# plt.axis('off')
# plt.title('Image with bbox and landmarks')
# plt.imshow(example_image)
# plt.show()

train_images = partitions.query("partition == 0")
valid_images = partitions.query("partition == 1")
test_images = partitions.query("partition == 2")

discriminator=keras.Sequential(
    [
        keras.Input(shape= (64, 64, 3)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size= 4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation= "sigmoid")
    ], name= "Discriminator"
)
discriminator.summary()


latent_dim = 100
generator = keras.Sequential([
    keras.Input(shape= (latent_dim, )),
    layers.Dense(8*8*128),
    layers.Reshape((8, 8, 128)),
    layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
    layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
], name="Generator"
)
generator.summary()

dataset = keras.preprocessing.image_dataset_from_directory(
    "/Users/himanshusharma/Documents/celeba/dataset1", label_mode=None, image_size=(64, 64), batch_size=32
)
dataset = dataset.map(lambda x: x / 255.0)

epochs=1000
gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer= keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer= keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=10, latent_dim=latent_dim)]

)



