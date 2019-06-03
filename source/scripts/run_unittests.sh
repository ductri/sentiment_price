# -*- coding: utf-8 -*-
export PYTHONPATH=`pwd`/main
set -e

echo "Running unit tests"

echo "Start running unit tests ..."
find main/ -type f -name *_test.py  |
while read filename
do
    echo 'Testing for: ' $(basename "$filename")
    echo "$filename"
    python "$filename"
done
echo "Done unit tests."
