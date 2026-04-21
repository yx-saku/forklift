#!/bin/bash -e

git clone https://github.com/cvat-ai/cvat -b v2.62.0 --depth 1

cd $(cd $(dirname $0); pwd)/cvat

# docker compose up -d
# docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

wget https://github.com/nuclio/nuclio/releases/download/1.15.9/nuctl-1.15.9-linux-amd64
sudo chmod +x nuctl-1.15.9-linux-amd64
sudo ln -sf $(pwd)/nuctl-1.15.9-linux-amd64 /usr/local/bin/nuctl
