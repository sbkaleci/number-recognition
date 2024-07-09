import joblib
from image_processing import process_image, crop_digits
import os 

crop_digits('./image.png')

images = os.listdir('./temp')
predictions = ''

for i, image in enumerate(images):
    path = os.path.join('./temp', image)
    arr = process_image(path)
    arr = arr.reshape(1, -1)
    model = joblib.load("model.pkl")
    prediction = model.predict(arr)
    predictions += prediction[0]
    os.remove(path)
    

print(predictions)