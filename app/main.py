#from typing import Union
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from fastai.vision.all import *

app = FastAPI()

origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## path = Path('./app/models') ## docker / heroku
path = Path('./models') ## local
file_path = path/"bird_classifier_resnet18_963.pkl"

learn = load_learner(file_path)
classes = learn.dls.vocab

def predict(img):
    label, _, probs = learn.predict(img)
    return {'label': label, 'confidences': zip(classes, list(map(float, probs)))}

@app.get("/")
def read_root():
    #label, probs = predict('./app/example.jpeg') ## docker / heroku
    #label, probs = predict('./example.jpeg') ## local

    results = predict('./examle.jpeg')
    ## return {"label": label, "confidences": probs}
    return JSONResponse(content=jsonable_encoder(result))
