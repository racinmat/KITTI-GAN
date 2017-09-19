#!/usr/bin/env bash

find ./checkpoint/$1 -type f -exec md5sum {} \; | sort -k 2 | md5sum
find ./logs/$1 -type f -exec md5sum {} \; | sort -k 2 | md5sum
find ./samples/$1 -type f -exec md5sum {} \; | sort -k 2 | md5sum
