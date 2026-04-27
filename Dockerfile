FROM python:3.11-slim

ARG USER_NAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    HOME=/home/${USER_NAME} \
    USER=${USER_NAME}

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-noto-cjk \
        libgomp1 \
        git \
        sudo \
        bubblewrap \
        npm \
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

RUN groupadd -g $USER_GID $USER_NAME && \
    useradd --create-home --home-dir /home/$USER_NAME --shell /bin/bash -u $USER_UID -g $USER_GID --groups adm,sudo $USER_NAME && \
    echo $USER_NAME:$USER_NAME | chpasswd && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN npm i -g @openai/codex@latest

EXPOSE 8888

USER ${USER_NAME}:${USER_NAME}

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token=forklift", "--PasswordIdentityProvider.hashed_password="]
