#!/bin/bash

process_variable=487245

while :
do
  cat "/proc/${process_variable}/status" | grep VmRSS
  cat "/proc/${process_variable}/status" | grep VmRSS >> mem-G-vs-N=14.txt
  sleep 5
done
