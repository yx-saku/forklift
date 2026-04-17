FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install \
        jupyterlab==4.4.1 \
        librosa==0.10.2.post1 \
        matplotlib==3.10.1 \
        numpy==2.2.4 \
        opencv-python-headless==4.11.0.86 \
        pandas==2.2.3 \
        scikit-learn==1.6.1 \
        scipy==1.15.2 \
        soundfile==0.13.1

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--IdentityProvider.token=forklift", "--PasswordIdentityProvider.hashed_password="]
