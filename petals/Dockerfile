FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

# Install Python
RUN apt-get update && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.12-pip python3.12-dev git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

RUN git clone https://github.com/petals-infra/chat.petals.dev.git
WORKDIR /usr/src/app/chat.petals.dev
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "flask", "run", "--host", "0.0.0.0" ]
EXPOSE 5000