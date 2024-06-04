FROM debian:12 as tesseract_base

ENV TESSDATA_PREFIX=$HOME/.local/share/tessdata_best
WORKDIR /app

# Install dependencies
RUN apt-get update \
 && apt-get install -y apt-utils autoconf libleptonica-dev libpango1.0-dev cabextract libarchive-dev libcurl4-openssl-dev libcurl4 curl libtool git g++ make \
 && git clone --depth 1 --branch 5.3.4 https://github.com/tesseract-ocr/tesseract.git /app
# Setup, configure, install
RUN mkdir -p m4 \
 && ./autogen.sh \
 && mkdir -p $HOME/local/lib && ./configure --disable-shared --enable-openmp --disable-doc CXX=g++ CXXFLAGS='-O2' --prefix=$HOME/local/ --with-extra-libraries=$HOME/local/lib \
 && make && make install

# RUN mkdir -p $HOME/.local/share \
# && git clone --depth 1 https://github.com/tesseract-ocr/tessdata_best.git $HOME/.local/share

RUN echo 'alias l="ls -l --color --group-directories-first"' >> ~/.bashrc

#CMD tesseract --images/page.png out_file --tessdata-dir $TESSDATA_PREFIX -l fra -psm 1

FROM python:3.11-bullseye

WORKDIR /app

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    --mount=type=cache,mode=0755,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY config /app/config
COPY src /app/src

CMD ["gunicorn", "--config", "config/gunicorn.conf.py", "--worker-tmp-dir", "/dev/shm", "--log-config", "config/logging.ini", "src.fastapi_app:app"]
