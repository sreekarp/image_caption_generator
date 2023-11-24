import io
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

templates = Jinja2Templates(directory="templates")
app = FastAPI()

# Clear any previous session
tf.keras.backend.clear_session()

# Load the Keras model
model = tf.keras.models.load_model('model.h5')

# Load the pickled features and tokenizer
#with open('features.pkl', 'rb') as f:
 #   features = pickle.load(f)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define the max length for captions
max_length = 35

# Generate a caption for an image
def predict_caption(model, image, tokenizer, max_length):
    in_text = ''  # Start the caption generation
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        if word == 'endseq':
            break
        in_text += ' ' + word
    return in_text

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index1.html", {"request": request})

@app.post('/process')
async def process(request: Request,image: UploadFile = Form(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((224, 224))  # Resize the image to (224, 224)
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    features = vgg_model.predict(img, verbose=0)
    caption = predict_caption(model, features, tokenizer, max_length)
    return templates.TemplateResponse("result1.html", {"request": request, "prediction": caption})
