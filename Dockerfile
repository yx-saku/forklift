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
        libgomp1 \
        git \
        sudo \
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

RUN set -eux; \
    if ! getent group "${USER_GID}" >/dev/null; then \
        groupadd --gid "${USER_GID}" "${USER_NAME}"; \
    fi; \
    if ! id --user "${USER_UID}" >/dev/null 2>&1; then \
        useradd \
            --uid "${USER_UID}" \
            --gid "${USER_GID}" \
            --create-home \
            --shell /bin/sh \
            "${USER_NAME}"; \
    fi; \
    mkdir -p /workspace "${HOME}"; \
    chown -R "${USER_UID}:${USER_GID}" /workspace "${HOME}"

EXPOSE 8888

USER ${USER_NAME}:${USER_NAME}

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token=forklift", "--PasswordIdentityProvider.hashed_password="]
