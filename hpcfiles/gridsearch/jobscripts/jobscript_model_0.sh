#!/bin/sh
### General options
### - specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J model_0_train
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request X GB of system-memory
#BSUB -R "rusage[mem=32GB]"
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo /zhome/04/d/118529/projects/procgenac/hpcfiles/.outfiles/0.out
#BSUB -eo /zhome/04/d/118529/projects/procgenac/hpcfiles/.errfiles/0.err
# -- end of LSF options --

# Load the cuda module
module load cuda/10.2

# -- run in the jobscripts directory --
cd ~/projects/procgenac/hpcfiles

source ~/miniconda3/bin/activate procgenac
python train.py \
    --model_type A2C \
    --env_name starpilot \
    --num_envs 32 \
    --num_levels 200 \
    --feature_dim 128 \
    --value_coef 0.5 \
    --entropy_coef 0.01 \
    --eps 0.2 \
    --grad_eps 0.5 \
    --num_epochs 3 \
    --batch_size 512 \
    --adam_lr 5e-5 \
    --adam_eps 1e-3 \
    --num_steps 256 \
    --total_steps 25_000_000 \
    --cnn_type impala \
    --model_id 0
