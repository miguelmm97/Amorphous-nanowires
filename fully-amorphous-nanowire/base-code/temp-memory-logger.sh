#!/bin/bash


while :
do
  cat /proc/1483404/status | grep VmRSS
  cat /proc/1483404/status | grep VmRSS >> mem-local-marker-Nx=12.txt
  sleep 5
done
