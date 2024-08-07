import tensorflow as tf
import os
import numpy as np

HEALTHY_DIR = '/Users/adam/Downloads/Potato___healthy/'
UNHEALTHY_DIR = '/Users/adam/Downloads/Potato___Late_blight/'

def load_images_and_labels(healthy_dir, unhealthy_dir):
    healthy_images = []
    unhealthy_images = []
    labels = []

    for image_file in os.listdir(healthy_dir):
        try:
            image = tf.keras.preprocessing.image.load_img(os.path.join(healthy_dir, image_file),
                                                          target_size=(256, 256))
            image = tf.keras.preprocessing.image.img_to_array(image)
            healthy_images.append(image)
            labels.append(0)
        except:
            pass

    for image_file in os.listdir(unhealthy_dir):
        try:
            image = tf.keras.preprocessing.image.load_img(os.path.join(unhealthy_dir, image_file),
                                                          target_size=(256, 256))
            image = tf.keras.preprocessing.image.img_to_array(image)
            unhealthy_images.append(image)
            labels.append(1)
        except:
            pass

    healthy_images = np.array(healthy_images)
    unhealthy_images = np.array(unhealthy_images)
    labels = np.array(labels)

    return healthy_images, unhealthy_images, labels


healthy_images, unhealthy_images, labels = load_images_and_labels(HEALTHY_DIR, UNHEALTHY_DIR)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(np.concatenate([healthy_images, unhealthy_images]), labels, epochs=10)

model.save('potato_model.h5')
