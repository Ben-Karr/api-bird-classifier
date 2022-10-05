FROM benkarr/fasta-p-i:1.1

WORKDIR /code

COPY ./app /code/app

CMD uvicorn app.main:app --host 0.0.0.0 --port=$PORT