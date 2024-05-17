FROM debian:12 as tesseract_base

ENV TESSDATA_PREFIX=$HOME/.local/share/tessdata_best

# Install dependencies
RUN apt-get update \
 && apt-get install -y autoconf libleptonica-dev libpango1.0-dev cabextract libarchive-dev libcurl4-openssl-dev libcurl4 curl libtool git
RUN mkdir -p /app/tesseract \
 && git clone --depth 1 --branch 5.3.4 https://github.com/tesseract-ocr/tesseract.git /app/tesseract
# RUN mkdir -p $HOME/.local/share \
# && git clone --depth 1 https://github.com/tesseract-ocr/tessdata_best.git $HOME/.local/share
RUN apt-get install -y g++

WORKDIR /app

# Setup, configure, install
RUN mkdir -p m4 \
 && ./autogen.sh \
 && ./configure --disable-shared --enable-openmp --disable-doc CXX=g++ CXXFLAGS='-O2' --prefix=/usr/local
RUN make \
 && make install
#
#CMD tesseract --images/page.png out_file --tessdata-dir $TESSDATA_PREFIX -l fra -psm 1

FROM python:3.11-bullseye

WORKDIR /app

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    --mount=type=cache,mode=0755,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY config /app/config
COPY src /app/src

CMD ["gunicorn", "--config", "config/gunicorn.conf.py", "--worker-tmp-dir", "/dev/shm", "--log-config", "config/logging.ini", "src.fastapi_app:app"]
