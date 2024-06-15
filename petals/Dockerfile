FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

# Install Python
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

RUN git clone https://github.com/petals-infra/chat.petals.dev.git
WORKDIR /usr/src/app/chat.petals.dev
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "flask", "run", "--host", "0.0.0.0" ]
EXPOSE 5000