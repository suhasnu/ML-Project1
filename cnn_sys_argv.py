import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def load_data(filename):
    
  data = np.load(filename)
  images = data['images']
  labels = data['labels']
  
  #Normalize (0-255 -> 0-1)
  images = images.astype('float32') / 255.
  
  #Reshape for CNN (add the 'channel' dimension)
  #Becomes (N, 28, 28, 1)
  images = np.expand_dims(images, axis=-1)
  
  X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42)
  
  return X_train, X_test, y_train, y_test


def build_model():
  model = tf.keras.Sequential([
    #Input: 28x28 image, 1 channel(grayscale)
    tf.keras.layers.Input(shape=(28,28,1)),
    
    #Block 1: Find simple edges
    tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    
    #Block 2: Find shapes
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    
    #Block 3: Find complex patterns
    tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    
    #Flatten: Turn the grid into a long list of numbers
    tf.keras.layers.Flatten(),
    
    #Dense layers: The "Thinking Part"
    tf.keras.layers.Dropout(0.4), #Prevent overfitting
    tf.keras.layers.Dense(128, activation="relu",),
    tf.keras.layers.Dropout(0.4),
    
    #Output: 10 probabilities (one for each clothing type)
    tf.keras.layers.Dense(10, activation="softmax"),
    
  ])
  
  
  #We use 'sparse_categorica_crossentropy' so we don't need to do One-Hot encoding!!
  model.compile( optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics=["accuracy"])
  
  return model

def main():
  #1.Check Arguments
  if len(sys.argv) < 3:
    sys.exit("Give enough arguments!")
    
  npz_file = sys.argv[1]
  mode = sys.argv[2]
  
  #2. Get Data
  X_train, X_test, y_train, y_test = load_data(npz_file)
  
  #Get Brain
  model = build_model()
  
  #4. Decision: Test or Train?
  if mode == "train":
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
    
    model.save("my_model.keras")
    print("Model saved!!")
    
  else:
    model.load_weights("my_model.keras")
    loss, acc = model.evaluate(X_test, y_test)
    
    print("Final Test Accuracy:", round(acc * 100, 2), "%")
    
    #Confusion Matrix
    prediction = model.predict(X_test)
    
    #Convert probablities to class numbers(e.g. [0.1, 0.9, 0.0] -> 1)
    pred_classes = np.argmax(prediction, axis=1)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred_classes))
    

if __name__ == "__main__":
  main()
    
