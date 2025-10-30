#!/bin/bash

#SBATCH --job-name=comp3710_task3   
#SBATCH --output=task3_job_%j.out   
#SBATCH --error=task3_job_%j.err   

#SBATCH --partition=comp3710    
#SBATCH --gres=gpu:1                
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1        
#SBATCH --cpus-per-task=4          
#SBATCH --mem=16G                   
#SBATCH --time=01:00:00            


module purge

module load cuda/11.1
module load anaconda3/2022.05

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710

python train.py
echo "Traning Complete, running predict.py ---"

python predict.py
echo "Task Finish"