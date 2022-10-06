# An API that uses Fastai and FastAPI to predict bird species

The app loads a fastai classification model trained [here](https://github.com/Ben-Karr/train-bird-classifier). You can post an image file to `"/"` to receive the predicted bird-label together with the overall per-label confidences.
The API is released as a Docker container on heroku which (in the free tier) has a strong limitation on available storage. In order to meet those storage limitations a particularly small model was used (`resnet18`@47MB) and the container builds from a prebuild image that contains all the necessary libraries (including a `torch` cpu installation). That `fasta-p-i` container can be found [here](https://hub.docker.com/repository/docker/benkarr/fasta-p-i) and is build with:

`Dockerfile`:
```Docker
FROM python:3.9-slim

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt
```
`requirements.txt`:
```
--find-links https://download.pytorch.org/whl/torch_stable.html
torch==1.11.0+cpu
--find-links https://download.pytorch.org/whl/torch_stable.html
torchvision==0.12.0+cpu
fastai
fastapi
uvicorn
python-multipart
```