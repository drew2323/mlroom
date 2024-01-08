#!/bin/bash

chmod 600 ~/.ssh/id_rsa
curl -X 'GET'   'http://5.161.179.223:8000/threads/'   -H 'accept: application/json'   -H 'X-API-Key: david' -v
sudo fallocate -l 30GB /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
pip install -e git+https://github.com/drew2323/mlroom.git#egg=mlroom
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --default-timeout=100
pip install git+https://github.com/drew2323/keras-tcn.git
pip install tqdm
pip install keras-nlp
pip install keras-nlp-nightly --default-timeout=100
pip install keras
cd src/mlroom/mlroom

#RUN ON LOCAL MAC (or other which has priv key) - transfer priv key 
scp ~/.ssh/id_ed25519 user@5.161.179.223:~/.ssh/

#PREP DATA TRANSFER

#TRANSFER TO PROD
#scp -c aes128-gcm@openssh.com training_data.joblib david@5.161.179.223:/mnt/HC_Volume_100253210
rsync -avz --progress training_data.joblib david@5.161.179.223:/mnt/HC_Volume_100253210/
rsync -avz --progress ~/src/mlroom/mlroom/training_data.joblib david@5.161.179.223:/mnt/HC_Volume_100253210/

#TRANSFER FROM PROD
#a naopak download prepared dat
#scp -c none david@5.161.179.223:/mnt/HC_Volume_100253210/training_data.joblib .
rsync -avz --progress david@5.161.179.223:/mnt/HC_Volume_100253210/training_data.joblib .
rsync -avz --progress david@5.161.179.223:/mnt/HC_Volume_100253210/training_data.joblib ~/src/mlroom/mlroom/

#
nohup python3 train.py > output.log 2>&1 &
2456

#RUN SCRIPTS
python3 train.py train --from_file training_data.joblib
nohup python3 train.py train --from_file training_data.joblib > output.log 2>&1 &

python3 train.py prepare --to_file training_data.joblib
nohup python3 train.py prepare --to_file training_data.joblib > output.log 2>&1 &

python3 train.py train --to_file training_data.joblib
nohup python3 train.py train --to_file training_data.joblib > output.log 2>&1 &

#--toml config.toml #config_tick.toml ...