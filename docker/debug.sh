#!/bin/bash

docker run --runtime=nvidia -ti --rm -e PYTHONIOENCODING=utf-8 --name="sentiment_price_debug" \
-v `pwd`/source:/source \
ductricse/pytorch /bin/bash -c "/source/scripts/train.sh"
