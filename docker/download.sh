#!/bin/bash

docker logs -f --timestamps $(docker run -d -e PYTHONIOENCODING=utf-8 --name="sentiment_price_$(date +"%y-%m-%d_%H_%M_%S")" \
-v `pwd`/source:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/download.sh")
