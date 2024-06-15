FROM nvidia/cuda:11.4.2-base-ubuntu20.04

# Install Python
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "serve.py" ]
EXPOSE 8765