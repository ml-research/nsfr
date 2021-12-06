#!/bin/sh

wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
mv CLEVR_v1.0 clevr
python preprocess-images.py
mkdir CLEVR
mv train-images.h5 CLEVR/
mv val-images.h5 CLEVR/
mv clevr/scenes CLEVR/
rm -r clevr
