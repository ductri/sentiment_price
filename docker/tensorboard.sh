#!/bin/bash

docker logs -f --timestamps $(docker run --rm -d -e PYTHONIOENCODING=utf-8 --name="sentiment_price_tensorboard" \
-v `pwd`/source:/source \
-p 10001:6006 \
ductricse/pytorch /bin/bash -c "/source/scripts/tensorboard.sh")
