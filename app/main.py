from typing import Union
from fastapi import FastAPI
from loguru import logger

from fastai.vision.all import *

app = FastAPI()

path = Path('./app/models')
file_path = path/"bird_classifier_resnet18_963.pkl"

logger.info(f'content at path: {path.ls()}')
logger.info(f'learner path: {file_path}')
logger.info(f'model size: {(file_path).stat()}')

learn = load_learner(file_path)
classes = learn.dls.vocab

def predict(img):
    label, _, probs = learn.predict(img)
    return label, zip(classes, list(map(float, probs)))
    #return 'robin', {'robin': 0.081, 'banana': 0.019}

@app.get("/")
def read_root():
    label, probs = predict('./app/example.jpeg')
    return {"label": label, "conidences": probs}