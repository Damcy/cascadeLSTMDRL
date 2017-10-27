#!/bin/bash

NOW_DIR=`dirname $0`
SRC_DIR=$NOW_DIR"/src"

cd $SRC_DIR

source ../setting/port_setting

python3 run.py $RECEIVE_PORT $SERVER_PORT "$SLOTS"
