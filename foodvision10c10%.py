# Food Vision -> 10 classes -> 10% data
# https://www.kaggle.com/datasets/dansbecker/food-101

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

import tensorflow as tf

from sklearn.metrics import confusion_matrix

# folders
train_folder = os.listdir('Machine Learning 3/FoodVision/10_food_classes_10_percent/10_food_classes_10_percent/train')
test_folder = os.listdir('Machine Learning 3/FoodVision/10_food_classes_10_percent/10_food_classes_10_percent/test')

labels_text = train_folder
labels_int = [i for i in range(len(labels_text))]

# get img_paths from each dir 
# training img_paths
train_img_paths = []
train_labels_text= []
train_labels_int = []
i = 0
for dir in train_folder:
    for each in os.listdir(f'Machine Learning 3/FoodVision/10_food_classes_10_percent/10_food_classes_10_percent/test/{dir}'):
        train_img_paths.append(f'Machine Learning 3/FoodVision/10_food_classes_10_percent/10_food_classes_10_percent/test/{dir}/{each}')
        train_labels_text.append(dir)
        train_labels_int.append(i)
    i += 1

# testing imgs
test_img_paths = []
test_labels_text= []
test_labels_int = []
i = 0
for dir in test_folder:
    for each in os.listdir(f'Machine Learning 3/FoodVision/10_food_classes_10_percent/10_food_classes_10_percent/train/{dir}'):
        test_img_paths.append(f'Machine Learning 3/FoodVision/10_food_classes_10_percent/10_food_classes_10_percent/train/{dir}/{each}')
        test_labels_text.append(dir)
        test_labels_int.append(i)
    i += 1

# preprocess paths into imgs
def preprocess(img):
    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    return img

# preprocess for data batches
def preprocess_for_batches(img, label):
    return preprocess(img), label

# create data batches
def create_data_batches(training=False, testing=False, batch_size=None, X_train=None, y_train=None, X_test=None, y_test=None):
    if training:
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(buffer_size=len(X_train))
        dataset = dataset.map(preprocess_for_batches)
        dataset = dataset.batch(batch_size=batch_size)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        dataset = dataset.shuffle(buffer_size=len(X_test))
        dataset = dataset.map(preprocess_for_batches)
        dataset = dataset.batch(batch_size=batch_size)
    return dataset

train_data_batch = create_data_batches(training=True, batch_size=32, X_train=train_img_paths, y_train=train_labels_int)
test_data_batch = create_data_batches(testing=True, batch_size=32, X_test=test_img_paths, y_test=test_labels_int)


# modelling : 
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        strides=1
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2,
        padding='valid'
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        strides=1
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2,
        padding='valid'
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        activation='relu',
        strides=1
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=2,
        padding='valid'
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=len(labels_text), activation='softmax')
])

cnn_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

cnn_model.fit(
    x=train_data_batch, 
    epochs=10,
    shuffle=True,
    verbose=2
)

preds = cnn_model.predict(test_data_batch)
preds_text = []
for pred in preds:
    preds_text.append(labels_text[np.argmax(pred)])
actuals = []
for img, label in test_data_batch.unbatch():
    actuals.append(labels_text[label])

fig, ax = plt.subplots()
ax.set(title='Confusion Matrix')
sns.heatmap(ax=ax, data=confusion_matrix(y_pred=preds_text, y_true=actuals), annot=True, xticklabels=labels_text, yticklabels=labels_text)
plt.show()
