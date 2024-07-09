
# Image Digit Recognition

This project processes images to recognize digits.

## Project Structure

- **main.py**: Main script to crop digits from an image, process them, and predict the digits.
- **image_processing.py**: Contains functions for processing and cropping digits from images.
- **number_recognition.ipynb**: Notebook used to create the digit recognition model.
- **mnist.ipynb**: Notebook containing experiments with the MNIST dataset.

## How to Use

1. **Install dependencies:**
   ```sh
   pip install pillow numpy opencv-python-headless joblib scikit-learn matplotlib
   ```

2. **Run the script:**
   Place the image named `image.png` in the root directory. Then execute:
   ```sh
   python main.py
   ```

## Notebooks

### number_recognition.ipynb
- **Purpose**: Create a digit recognition model.
- **Libraries**: scikit-learn, matplotlib, joblib, numpy

### mnist.ipynb
- **Purpose**: Experiment with the MNIST dataset.
- **Libraries**: scikit-learn, PIL, numpy, joblib

## Functionality

1. **Crop Digits:** 
   - Extracts individual digits from `image.png` and saves them in the `temp` directory.

2. **Process Image:**
   - Processes each cropped digit to prepare it for prediction.

3. **Predict Digits:**
   - Loads a pre-trained model (`model.pkl`) to predict the digits from the processed images and prints the result.
