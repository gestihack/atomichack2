FROM python:3.12

WORKDIR /usr/src/app

RUN git clone https://github.com/petals-infra/chat.petals.dev.git
WORKDIR /usr/src/app/chat.petals.dev
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "flask", "run", "--host", "0.0.0.0" ]
EXPOSE 5000