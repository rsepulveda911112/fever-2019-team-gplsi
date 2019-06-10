FROM python:3.6

RUN mkdir /fever

WORKDIR /fever

RUN wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip &&\
    mkdir -p data/glove&&\
    unzip glove.6B.zip -d data/glove &&\
    gzip data/glove/*.txt &&\
    rm glove.6B.zip

RUN wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip &&\
    mkdir -p data/fasttext &&\
    unzip wiki.en.zip -d data/fasttext &&\
    rm wiki.en.zip

RUN mkdir -p data/fever &&\
    wget -O data/fever/fever.db 'http://192.168.100.16:8080/fever.db'

RUN pip install Cython==0.28.5
RUN pip install numpy==1.14.5

ADD myrequiments.txt .

RUN pip install -r myrequiments.txt
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download en_core_web_sm

RUN python -c "import nltk; nltk.download('punkt')"

COPY . .

ENV PYTHONPATH=src
EXPOSE 5000

CMD ["python", "src/api.py"]
