#!/bin/bash -e

ls -la  ~

cd ~/cache

if [ ! -d ./venv ]; then
    python -m venv .venv
    . .venv/bin/activate

    python -m pip install -U pip
    python -m pip install \
        jupyterlab==4.4.1 \
        librosa==0.10.2.post1 \
        matplotlib==3.10.1 \
        numpy==2.2.4 \
        opencv-python-headless==4.11.0.86 \
        pandas==2.2.3 \
        scikit-learn==1.6.1 \
        scipy==1.15.2 \
        soundfile==0.13.1 \
        ipykernel==6.30.1 jupyter jupyter_client pyzmq tornado

    python -m ipykernel install --user \
    --name forklift \
    --display-name "Python (forklift)"
else
    . .venv/bin/activate
fi

exec "$@"