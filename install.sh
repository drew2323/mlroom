#!/bin/bash

# Creating a swap file
sudo fallocate -l 15GB /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Installing dependencies
pip install -e git+https://github.com/drew2323/mlroom.git#egg=mlroom
#pip install tf-nightly
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install git+https://github.com/drew2323/keras-tcn.git
pip install tqdm
#temporarily
pip install —U keras-nlp
pip install —U keras
#pip install —upgrade keras-nlp-nightly

# Navigate to the mlroom directory
cd src/mlroom/mlroom

curl -X 'GET'   'http://5.161.179.223:8000/threads/'   -H 'accept: application/json'   -H 'X-API-Key: david' -v
sudo fallocate -l 16GB /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
pip install -e git+https://github.com/drew2323/mlroom.git#egg=mlroom
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install git+https://github.com/drew2323/keras-tcn.git
pip install tqdm

#pip install —U keras-nlp

pip install —-upgrade keras-nlp-nightly
pip install —U keras

# Navigate to the mlroom directory
cd src/mlroom/mlroom