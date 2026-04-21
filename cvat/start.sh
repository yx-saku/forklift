#!/bin/bash -e

cd $(cd $(dirname $0); pwd)/cvat

# docker compose down
# docker compose up -d

docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d