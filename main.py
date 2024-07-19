from fastapi import FastAPI, File, UploadFile
from tensorflow.keras import utils
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import io
import uvicorn


cnn = load_model('maize_disease.h5')


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_headers = ['*'],
    allow_credentials = True,
    allow_methods = ['*'],
    allow_origins = ['*']
)


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image_data = io.BytesIO(image_data)
    test_image = utils.load_img(image_data, target_size = (64, 64))
    test_image = utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image)
    predict = np.argmax(result, axis = 1)
    if predict == 0:
        prediction = 'Your crops are Bilght'
    elif predict == 1:
        prediction = 'Your crops have Common Rust'
    elif predict == 2:
        prediction = 'Your crops have a Gray Leaf spot'
    else:
        prediction = 'Your crops are Healthy'
    
    return prediction
if __name__ == '__main__':
    uvicorn.run(app)