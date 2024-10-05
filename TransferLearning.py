import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
PATH = "C:\\Users\\29459\\Desktop\\dataset"

# train_dir = os.path.join(PATH, 'train')
# validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(PATH,
                                                                    validation_split=0.2,
                                                                    subset = "training",
                                                                    # shuffle=True,
                                                                    seed = 123,
                                                                    batch_size=BATCH_SIZE,
                                                                    image_size=IMG_SIZE)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(PATH,
                                                                         validation_split=0.2,
                                                                         subset = "validation",
                                                                         seed = 123,
                                                                         # shuffle=True,
                                                                         batch_size=BATCH_SIZE,
                                                                         image_size=IMG_SIZE)

class_names = train_dataset.class_names

# val_batches = tf.data.experimental.cardinality(validation_dataset)
# test_dataset = validation_dataset.take(val_batches // 5)
# validation_dataset = validation_dataset.skip(val_batches // 5)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
# test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False
# base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


initial_epochs = 10
history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# 第一个子图
axs[0].plot(acc, label='Training Accuracy')
axs[0].plot(val_acc, label='Validation Accuracy')
axs[0].legend(loc='lower right')
axs[0].set_ylabel('Accuracy')
axs[0].set_ylim([min(axs[0].get_ylim()), 1])
axs[0].set_title('Training and Validation Accuracy')

# 第二个子图
axs[1].plot(loss, label='Training Loss')
axs[1].plot(val_loss, label='Validation Loss')
axs[1].legend(loc='upper right')
axs[1].set_ylabel('Cross Entropy')
axs[1].set_ylim([0, 1.0])
axs[1].set_title('Training and Validation Loss')
axs[1].set_xlabel('epoch')

plt.tight_layout()  # 自动调整子图布局
plt.show()

# =========================保存模型================================

model.save('DemoWithoutFT.keras')

# ===============================================================


base_model.trainable = True
# print("Number of layers in the base model: ", len(base_model.layers))
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])
# model.summary()

fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs
history_fine = model.fit(train_dataset, epochs=total_epochs, initial_epoch=history.epoch[-1],
                          validation_data=validation_dataset)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']


fig, axs = plt.subplots(2, 1, figsize=(8, 8))

# 第一个子图
axs[0].plot(acc, label='Training Accuracy')
axs[0].plot(val_acc, label='Validation Accuracy')
axs[0].set_ylim([0.8, 1])
axs[0].set_title('Training and Validation Accuracy')
axs[0].legend(loc='lower right')
axs[0].plot([initial_epochs-1, initial_epochs-1], axs[0].get_ylim(), label='Start fine tuning')

# 第二个子图
axs[1].plot(loss, label='Training Loss')
axs[1].plot(val_loss, label='Validation Loss')
axs[1].set_ylim([0, 1.0])
axs[1].plot([initial_epochs-1, initial_epochs-1], axs[1].get_ylim(), label='Start fine tuning')
axs[1].legend(loc='upper right')
axs[1].set_title('Training and Validation Loss')
axs[1].set_xlabel('epoch')

plt.tight_layout()  # 自动调整子图布局
plt.show()



# =========================保存模型================================

model.save('DemoWithFT.keras')

# ===============================================================