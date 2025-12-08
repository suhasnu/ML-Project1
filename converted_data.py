import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image
import matplotlib.image as mpimg


#Method 1
image_pil = Image.open("0-0.png")
image_array = np.array(image_pil)

print(image_array.shape)
 
# Method 2
image_array = mpimg.imread("0-0.png")

print(image_array.shape)

plt.imshow(image_array)
plt.show()

image_array1 = mpimg.imread("9-54104.png")
print(image_array1.shape)

plt.imshow(image_array1)
plt.show()

#Step 1
