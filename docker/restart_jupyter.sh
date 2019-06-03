#!/bin/bash

docker stop sentiment_price
sleep 5

./docker/jupyter.sh
