FROM tiangolo/kigstn/uvicorn-gunicorn-fastapi:python3.9.7

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

