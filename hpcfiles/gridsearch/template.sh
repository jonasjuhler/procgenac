#!/bin/sh
### General options
### - specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J model_<model_id>_train
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
#BSUB -oo /zhome/04/d/118529/projects/procgenac/hpcfiles/.outfiles/<model_id>.out
#BSUB -eo /zhome/04/d/118529/projects/procgenac/hpcfiles/.errfiles/<model_id>.err
# -- end of LSF options --

# Load the cuda module
module load cuda/10.2

# -- run in the jobscripts directory --
cd ~/projects/procgenac/hpcfiles

source ~/miniconda3/bin/activate procgenac
python train.py \
    --model_type <model_type> \
    --env_name <env_name> \
    --num_envs <num_envs> \
    --num_levels <num_levels> \
    --feature_dim <feature_dim> \
    --value_coef <value_coef> \
    --entropy_coef <entropy_coef> \
    --eps <eps> \
    --grad_eps <grad_eps> \
    --num_epochs <num_epochs> \
    --batch_size <batch_size> \
    --adam_lr <adam_lr> \
    --adam_eps <adam_eps> \
    --num_steps <num_steps> \
    --total_steps <total_steps> \
    --cnn_type <cnn_type> \
    --model_id <model_id>
