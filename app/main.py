from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware
import io

from fastai.vision.all import *

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://ben-karr.github.io"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path = Path('./app/models') ## docker / heroku
##path = Path('./models') ## local
file_path = path/"bird_classifier_resnet18_963.pkl"

learn = load_learner(file_path)
classes = learn.dls.vocab

def predict(file):
    img = np.array(Image.open(io.BytesIO(file)))
    label, _, probs = learn.predict(img)
    return {'label': label, 'confidences': zip(classes, list(map(float, probs)))}

@app.post("/")
def read_root(file: bytes = File(...)):
    return predict(file)