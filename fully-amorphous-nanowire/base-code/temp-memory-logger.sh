#!/bin/bash

process_variable=893986

while :
do
  cat "/proc/${process_variable}/status" | grep VmRSS
  # cat "/proc/${process_variable}/status" | grep VmRSS >> mem-marker-KPM-15.txt
  sleep 5
done
