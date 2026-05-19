#!/bin/bash
set -xe

while true
do
    sleep 3
    if [ $(ps -ef |grep "inference/run_.*.py" |wc -l) -lt 2 ];then
        break
    fi
    echo "$(date +%s), $(numastat -p $(ps -ef |grep "inference/run_.*.py" |grep -v grep |awk '{printf("%s  ", $2)}'))"
done

