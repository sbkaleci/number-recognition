from PIL import Image, ImageOps
import numpy as np
import cv2
import os


def process_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = np.array(image)
    image = (image > 128).astype(np.uint8) * 255
    image = 255 - image

    white_rows = np.all(image == 0, axis=1)
    white_cols = np.all(image == 0, axis=0)

    image = image[~white_rows, :]
    image = image[:, ~white_cols]

    greater = max(image.shape)
    less = min(image.shape)

    greater_padding = round(greater * 1/2)
    greater_padding = round(greater_padding / 2)
    greater_padding = (int(greater_padding), int(greater_padding))

    side_length = greater + greater_padding[0] * 2

    less_padding = side_length - less

    if less_padding % 2 == 1:
        less_padding = (int((less_padding - 1) / 2),
                        int((less_padding + 1) / 2))
    else:
        less_padding = (int(less_padding / 2), int(less_padding / 2))

    if greater == image.shape[0]:
        image = np.pad(image, pad_width=(greater_padding,
                       less_padding), mode='constant', constant_values=0)
    else:
        image = np.pad(image, pad_width=(
            less_padding, greater_padding), mode='constant', constant_values=0)

    image = Image.fromarray(image)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image = np.array(image)

    shifted_image = np.roll(image, shift=1, axis=1)
    image = np.maximum(shifted_image, image)

    shifted_image = np.roll(image, shift=1, axis=0)
    image = np.maximum(shifted_image, image)

    image = image.astype(np.uint16)
    image[image > 20] += 64
    image[image >= 255] = 255
    image = image.astype(np.uint8)
    image = image.flatten()

    return image


def crop_digits(image_path):
    image = Image.open(image_path)
    gray = ImageOps.grayscale(image)
    gray_np = np.array(gray)
    blurred = cv2.GaussianBlur(gray_np, (5, 5), 0)
    _, binary_np = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(binary_np, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_contour_area = 50
    max_contour_area = gray_np.shape[0] * gray_np.shape[1] / 2
    valid_contours = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        contour_area = cv2.contourArea(contour)
        
        if min_contour_area < contour_area < max_contour_area and 0.2 < aspect_ratio < 1.0:
            valid_contours.append((x, y, w, h))
    
    valid_contours = sorted(valid_contours, key=lambda rect: rect[0])
    output_dir = './temp'
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (x, y, w, h) in enumerate(valid_contours):
        digit = image.crop((x, y, x + w, y + h))
        digit.save(os.path.join(output_dir, f'digit_{i}.png'))
