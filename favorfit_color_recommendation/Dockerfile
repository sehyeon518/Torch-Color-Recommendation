FROM python:3.8

WORKDIR /shlee

RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev
    
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY ./lambda_function.py ./
COPY ./features ./features
COPY ./models/inference.py ./models/inference.py
COPY ./models/model.py ./models/model.py
COPY ./models/20240124.json ./models/20240124.json
COPY ./features/ ./features/
COPY ./utils/ ./utils/

ENTRYPOINT [ "python", "lambda_function.py" ]

# RUN cp ./lambda_function.py /shlee/