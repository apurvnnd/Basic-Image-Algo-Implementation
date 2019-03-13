
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from PIL import Image, ImageDraw
from math import sqrt
from scipy import misc
get_ipython().magic('matplotlib inline')


# In[8]:

def brightness(inputfile,outputfile,brightness):
    with open(inputfile,'rb') as f:
        img_byte = f.read()
    #     print(img_byte)
        img = np.asarray(Image.open(io.BytesIO(img_byte)))
        img.setflags(write=1)
        for i in range(len(img)):
            for j in range(len(img[i])):
                for k in range(3):
                    if img[i][j][k]*brightness<255.0:
                        img[i][j][k] = img[i][j][k]*brightness
                    else:
                        img[i][j][k] = 255.0
        misc.imsave(outputfile, img)
        return 'Success'


# In[11]:

# Brightness
brightness('princeton_small.jpg','princeton_small_brightness_2.0.jpg',2.0)


# In[12]:

#Contrasting value 
def set_contrast(rgb,contrast):
    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
    c1 = 128 + contrast * (rgb - 128)
    if c1>255:
        c1 = 255
    elif c1<255 and c1>0:
        c1 = 128 + contrast * (rgb - 128)
    elif c1<0:
        c1 = 0
    return c1


# In[14]:

def contrast_op(inputfile,outputfile,cont):
    with open(inputfile,'rb') as f:
        img_byte = f.read()
    #     print(img_byte)
        img = np.asarray(Image.open(io.BytesIO(img_byte)))
        img.setflags(write=1)
        for i in range(len(img)):
            for j in range(len(img[i])):
                for k in range(3):
                    img[i][j][k] = set_contrast(img[i][j][k],contrast=cont)

        misc.imsave(outputfile, img)
        return 'Success'


# In[18]:

contrast_op('c.jpg','c_contrast_2.0.jpg',2.0)


# In[21]:

def sharpness(inputfile,outputfile):
    # Load image:
    input_image = Image.open(inputfile)
    input_pixels = input_image.load()

    # High-pass kernel
    kernel = [[  0  , -.5 ,    0 ],
              [-.5 ,   3  , -.5 ],
              [  0  , -.5 ,    0 ]]

    # Middle of the kernel
    offset = len(kernel) // 2

    # Create output image
    output_image = Image.new("RGB", input_image.size)
    draw = ImageDraw.Draw(output_image)

    # Compute convolution with kernel
    for x in range(offset, input_image.width - offset):
        for y in range(offset, input_image.height - offset):
            acc = [0, 0, 0]
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = x + a - offset
                    yn = y + b - offset
                    pixel = input_pixels[xn, yn]
                    acc[0] += pixel[0] * kernel[a][b]
                    acc[1] += pixel[1] * kernel[a][b]
                    acc[2] += pixel[2] * kernel[a][b]

            draw.point((x, y), (int(acc[0]), int(acc[1]), int(acc[2])))

    output_image.save(outputfile)
    return 'Success'


# In[23]:

sharpness('princeton_small.jpg','sharpen.jpg')


# In[24]:

def edge(inputfile,outputfile):
    # Load image:
    input_image = Image.open(inputfile)
    input_pixels = input_image.load()

    # Calculate pixel intensity as the average of red, green and blue colors.
    intensity = [[sum(input_pixels[x, y]) / 3 for y in range(input_image.height)] for x in range(input_image.width)]

    # Sobel kernels
    kernelx = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    kernely = [[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]]

    # Create output image
    output_image = Image.new("RGB", input_image.size)
    draw = ImageDraw.Draw(output_image)

    # Compute convolution between intensity and kernels
    for x in range(1, input_image.width - 1):
        for y in range(1, input_image.height - 1):
            magx, magy = 0, 0
            for a in range(3):
                for b in range(3):
                    xn = x + a - 1
                    yn = y + b - 1
                    magx += intensity[xn][yn] * kernelx[a][b]
                    magy += intensity[xn][yn] * kernely[a][b]

            # Draw in black and white the magnitude
            color = int(sqrt(magx**2 + magy**2))
            draw.point((x, y), (color, color, color))

    output_image.save(outputfile)
    return 'Success'


# In[25]:

edge('princeton_small.jpg','edgedetect.jpg')


# In[ ]:



