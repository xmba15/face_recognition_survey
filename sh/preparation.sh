#!/bin/bash

os_system=`uname`
if [ $os_system == "Darwin" ]; then
    realpath() {
        path=`eval echo "$1"`
        folder=$(dirname "$path")
        echo $(cd "$folder"; pwd)/$(basename "$path");
    }
fi

absolute_path=`pwd`/`dirname $0`

cd $absolute_path/../thirdparty/embedding/insightface/sh
./preparation.sh

cd $absolute_path/../thirdparty/pose/deep_head_pose/sh
./preparation.sh
