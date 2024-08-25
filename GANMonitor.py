import keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt


class GANMonitor(keras.callbacks.Callback):


    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        # Define the folder path where you want to save the generated images
        save_folder = '/Users/himanshusharma/Documents/celeba/dataset1/output_images'
        os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()

        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            # Save the image to the specified folder
            img_path = os.path.join(save_folder, "generated_img_%03d_%d.png" % (epoch, i))
            img.save(img_path)
            print(f"Image saved to {img_path}")