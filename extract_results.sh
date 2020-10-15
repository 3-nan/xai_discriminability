#!/bin/bash

for filename in results/vgg16/cifar10/LRPAlpha1Beta0/*.tgz
do
  echo $filename
  name=$(echo "$filename" | cut -f 1 -d '-')
  mkdir "$name"
  tar -C "$name"/ -xf $filename
done
