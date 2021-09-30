'''NEURAL STYLE TRANSFER'''


"""##Importing Libraries"""


import gradio as gr
# import tensorflow_hub as hub
import tensorflow as tf
# import os
import PIL
from PIL import Image,ImageOps
import numpy as np
# import time
# import requests   
import cv2
from cv2 import *

# !mkdir nstmodel 
# !wget -c https://storage.googleapis.com/tfhub-modules/google/magenta/arbitrary-image-stylization-v1-256/2.tar.gz -O - | tar -xz -C /nstmodel
# import tensorflow.keras

# from PIL import Image, ImageOps
import requests
import tarfile

url = "https://storage.googleapis.com/tfhub-modules/google/magenta/arbitrary-image-stylization-v1-256/2.tar.gz"
response = requests.get(url, stream=True)
file = tarfile.open(fileobj=response.raw, mode="r|gz")
file.extractall(path="./nst_model")

MODEL_PATH='./nst_model'

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

"""##Saving unscaled Tensor images."""

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save to.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)

"""## Grayscaling image for testing purpose to check if we could get better results."""



def gray_scaled(inp_img):
  gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
  gray_img = np.zeros_like(inp_img)
  gray_img[:,:,0] = gray
  gray_img[:,:,1] = gray
  gray_img[:,:,2] = gray
  return gray_img

def transform_mymodel(content_image,style_image):
  # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]
  content_image=gray_scaled(content_image)
  content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.0
  style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.0
 
  #Resizing image
  style_image = tf.image.resize(style_image, (256, 256))

  # Stylize image
  outputs = model(tf.constant(content_image), tf.constant(style_image))
  stylized_image = outputs[0]

  # stylized = tf.image.resize(stylized_image, (356, 356))
  stylized_image =tensor_to_image(stylized_image)
  save_image(stylized_image,'stylized')
  return stylized_image

def gradio_intrface(mymodel):
# Initializing the input component 
  image1 = gr.inputs.Image() #CONTENT IMAGE
  image2 = gr.inputs.Image() #STYLE IMAGE
  stylizedimg=gr.outputs.Image() 
  gr.Interface(fn=mymodel, inputs= [image1,image2] , outputs= stylizedimg,title='Style Transfer').launch()

"""The function will be launched both  Inline and Outline  where u need to add a content and style image."""


gradio_intrface(transform_mymodel)



