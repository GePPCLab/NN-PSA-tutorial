#!/usr/bin/env bash

DATA_URL=https://germlab.ca/data/library_noise_data.zip
ZIP_NAME=data.zip

mkdir -p data

wget $DATA_URL -O data/$ZIP_NAME

cd data/
unzip $ZIP_NAME
rm $ZIP_NAME
