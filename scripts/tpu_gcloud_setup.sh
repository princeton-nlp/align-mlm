#!/bin/bash

function SourceCodeAndInstall {
    mkdir source_code
    cd source_code
    git clone https://github.com/henrytang1/MultilingualModelAnalysis.git
    conda env list
    conda create --name multilingual --clone torch-xla-1.11
    conda activate multilingual
    cd MultilingualModelAnalysis/transformers/
    # In some instance, just `conda activate base` should work.
    # `import torch_xla` to check if it's the correct environment.
    pip install wandb
    pip install -e .
    pip install -r examples/language-modeling/requirements.txt
    pip install -r examples/token-classification/requirements.txt
}

function GCFuse {
    # Install gcsfuse
    export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
    echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    printf 'Y' | sudo apt-get install gcsfuse
    sudo apt-get install htop
}

function MountBucket {
    # Mount the bucket for saving model files
    cd ~
    mkdir bucket
    gcsfuse --implicit-dirs multilingual-1  bucket/
    cd source_code/MultilingualModelAnalysis/
}

function MakeTPUs {
    export VERSION=1.11
    gcloud compute tpus create h-tpu-1 --zone=us-central1-a --network=default --version=pytorch-1.11 --accelerator-type=v3-8
    gcloud compute tpus list --zone=us-central1-a
    export TPU_IP_ADDRESS=10.23.248.106	
    export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
}

# TPU List
############### h-exp-1 ###############
# export TPU_IP_ADDRESS=10.23.248.106	
############### h-exp-2 ###############
# export TPU_IP_ADDRESS=10.56.28.98
############### h-exp-3 ###############
# export TPU_IP_ADDRESS=10.28.229.122
############### h-exp-4 ###############
# export TPU_IP_ADDRESS=10.123.235.234	
############### h-exp-5 ###############
# export TPU_IP_ADDRESS=10.86.170.138 # USED FOR MLM #
############### h-exp-6 ###############
# export TPU_IP_ADDRESS=10.127.27.178
############### h-exp-7 ###############
# export TPU_IP_ADDRESS=10.8.124.90
############### h-exp-8 ###############
# export TPU_IP_ADDRESS=10.45.131.210	
############### h-exp-9 ###############
# export TPU_IP_ADDRESS=10.83.156.66
############### h-exp-10 ###############
# export TPU_IP_ADDRESS=10.65.182.186

function RestartVM {
    conda activate multilingual
    gcsfuse --implicit-dirs --debug_fuse multilingual-1  bucket/
    export VERSION=1.11
    gcloud compute tpus list --zone=us-central1-a
    export TPU_IP_ADDRESS=10.23.248.106	
    export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP_ADDRESS:8470"
    cd source_code/MultilingualModelAnalysis/
    git pull
    export WANDB_API_KEY="X"
    export WANDB_ENTITY="henrytang"
    export WANDB_PROJECT="tlm_multilingual_synthetic"
}

function Wandb {
    # wandb login
    export WANDB_API_KEY="X"
    export WANDB_ENTITY="henrytang"
    export WANDB_PROJECT="tlm_multilingual_synthetic"
    # export WANDB_NAME="wikitext_mlm"
    # Run name is specified using the --run_name argument
}

function GitHub {
  # If you want your git credentials to be stored in plain text
  git config --global credential.helper store
}

function UbuntuVM {
  sudo apt install git-all
  sudo apt-get install wget
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  sudo apt install default-jre
}

###

for arg in "$@"; do
  if [[ "$arg" = -i ]] || [[ "$arg" = --initial ]]; then
    ARG_INITIAL=true
    ARG_RESTART=false
  fi
  if [[ "$arg" = -r ]] || [[ "$arg" = --restart ]]; then
    ARG_RESTART=true
    ARG_INITIAL=false
  fi
done

###

if [[ "$ARG_INITIAL" = true ]]; then
  SourceCodeAndInstall
  GCFuse
  MountBucket
  Wandb
fi

if [[ "$ARG_RESTART" = true ]]; then
  RestartVM
  Wandb
fi