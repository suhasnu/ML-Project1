import tensorflow as tf
import numpy as np 


#loading the cleaned data
data = np.load("cleaned_data.npz")
images = data["images"]
labels = data["labels"]

#number of images
number_of_images = images.shape[0]

#indices are the images numers in shuffled order
indices = np.random.permutation(number_of_images)

train_size = int(0.8*number_of_images)

#taking the first 80% of the shuffeled images indices as train_data
train_ids = indices[:train_size]
#taking the last 20% of the shuffeled images indices as train_data
test_ids = indices[train_size:]

#train_set
train_set = images[train_ids]
train_lables =labels[train_ids]
#test_set
test_set = images[test_ids]
test_lables = labels[test_ids]


#turning labels into one hot format
train_labels_oh = tf.keras.utils.to_categorical(train_lables, 10)
test_labels_oh = tf.keras.utils.to_categorical(test_lables,10)

#number of output classes
num_classes = 10

#model training
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D((2, 2)),

    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D((2, 2)),

    # Block 3 
    tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),          # gegen Overfitting
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

opt    = tf.keras.optimizers.Adam(learning_rate=0.001)
loss   = tf.keras.losses.CategoricalCrossentropy()
metric = tf.keras.metrics.CategoricalAccuracy()

model.compile(
    optimizer=opt,
    loss=loss,
    metrics=[metric],
)

history = model.fit(
    train_set, train_labels_oh,
    epochs=15,
    batch_size=32,
    validation_data=(test_set, test_labels_oh)
)

