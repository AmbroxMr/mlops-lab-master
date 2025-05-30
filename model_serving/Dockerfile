FROM python:3.10.7

WORKDIR /home
COPY ./TextAIWrapper.py /home
COPY ./requirements.txt /home
RUN mkdir -p -v /home/files/tmp
RUN chown -R 8888 /home
RUN chmod -R 755 /home

RUN apt-get -y  update
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet

EXPOSE 5000
EXPOSE 9000

EXPOSE 6000

ENV MODEL_NAME="TextAIWrapper"
ENV SERVICE_TYPE="MODEL"

ENV MLFLOW_TRACKING_URI="http://host.docker.internal:32000"

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE
