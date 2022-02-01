#!/bin/bash
sudo yum update -y
sudo yum install python3 -y

python3 -m venv clustering
source clustering/bin/activate

pip install --upgrade pip
