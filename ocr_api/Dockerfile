FROM debian:12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ autoconf automake libtool pkg-config \
 && apt-get install -y --no-install-recommends libpng-dev libjpeg8-dev libtiff5-dev zlib1g-dev libwebpdemux2 libwebp-dev libopenjp2-7-dev libgif-dev libarchive-dev libcurl4-openssl-dev

RUN apt-get install -y --no-install-recommends libleptonica-dev tesseract-ocr
# && pip install --disable-pip-version-check --no-cache-dir -r requirements.txt \
#RUN apt-get purge -y gcc && apt-get autoremove -y \
# && rm -rf /var/lib/apt/lists/* \





