#!/bin/bash -e

set -euo pipefail

USER_NAME="${USER_NAME:-${USER:-appuser}}"
export HOME="${HOME:-/home/${USER_NAME}}"

if ! mkdir -p "${HOME}" 2>/dev/null; then
  export HOME="/tmp/${USER_NAME}"
fi

VENV_DIR="${HOME}/cache/.venv"
VENV_SETUP_MARKER="${VENV_DIR}/.bootstrap_cpu_done"

mkdir -p "${HOME}/.cache" "${HOME}/.config" "${HOME}/.local/share" "${HOME}/cache"

if command -v sudo >/dev/null 2>&1; then
  sudo chown "$(id -u):$(id -g)" -R "${HOME}/.config" "${HOME}/.local" "${HOME}/cache" || true
fi

if [ -d "${VENV_DIR}" ] && [ ! -f "${VENV_SETUP_MARKER}" ]; then
  echo "[entrypoint] removing stale or incomplete venv: ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  python3.11 -m venv "${VENV_DIR}"
fi

. "${VENV_DIR}/bin/activate"

if [ ! -f "${VENV_SETUP_MARKER}" ]; then
  python -m pip install --upgrade pip setuptools wheel \
     && python -m pip install \
         --no-cache-dir \
         jupyterlab==4.4.1 \
         librosa==0.10.2.post1 \
         matplotlib==3.10.1 \
         numpy==2.2.4 \
         opencv-python-headless==4.11.0.86 \
         pandas==2.2.3 \
         scikit-learn==1.6.1 \
         scipy==1.15.2 \
         soundfile==0.13.1 \
         joblib==1.4.2 \
         tqdm==4.67.1 \
         ipykernel==6.30.1 \
         jupyter \
         jupyter_client \
         pyzmq \
         tornado
  touch "${VENV_SETUP_MARKER}"
fi

python -m ipykernel install --user --name forklift --display-name "Python (forklift)"

exec "$@"
