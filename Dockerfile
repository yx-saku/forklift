FROM python:3.11-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USER_NAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/${USER_NAME} \
    USER=${USER_NAME}

WORKDIR /workspace

RUN apt-get update \
     && apt-get install -y --no-install-recommends \
         ffmpeg \
         git \
         sudo \
     && rm -rf /var/lib/apt/lists/*

RUN groupadd -g $USER_GID $USER_NAME && \
    useradd --create-home --home-dir /home/$USER_NAME --shell /bin/bash -u $USER_UID -g $USER_GID --groups adm,sudo $USER_NAME && \
    echo $USER_NAME:$USER_NAME | chpasswd && \
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8888

USER ${USER_NAME}

ENTRYPOINT ["/entrypoint.sh"]

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token=forklift", "--PasswordIdentityProvider.hashed_password="]
