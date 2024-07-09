#convert to greyscale
#treshold to reduce noice
#remove the empty row and columns
#complete image to a square based on size of number (eg. 80% of image wil consist of number)
#resize the image the image to 28x28
#convert image to an array of 784 images
#send image to model

from PIL import Image
import numpy as np
from scipy.ndimage import zoom


def process_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.array(image)
    image = (image < 128).astype(np.uint8)

    white_rows = np.all(image == 255, axis=1)
    white_cols = np.all(image == 255, axis=0)

    image = image[~white_rows, :]
    image = image[:, ~white_cols]

    greater = max(image.shape)
    less = min(image.shape)
    
    greater_padding = round(greater * 5/4)
    greater_padding = round(greater_padding / 2)
    greater_padding = (int(greater_padding), int(greater_padding))

    side_length = greater + greater_padding[0] * 2
    
    less_padding = side_length - less
    
    if less_padding % 2 == 1:
        less_padding = (int((less_padding - 1) / 2), int((less_padding + 1) / 2))
    else:
        less_padding = (int(less_padding / 2), int(less_padding / 2)) 

    print(greater_padding, less_padding)

    if greater == image.shape[0]:
        image = np.pad(image, pad_width=(less_padding, greater_padding), mode='constant', constant_values=255)
    else:
        image = np.pad(image, pad_width=(greater_padding, less_padding), mode='constant', constant_values=255)

    image = Image.fromarray(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = np.array(image)

    img = Image.fromarray(image.astype(np.uint8))
    img.save('resized_image.png')

    image = image.flatten()
    image = np.divide(image, 255)

    return image