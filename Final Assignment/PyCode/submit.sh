#!/bin/bash
#PBS -N BayesianLearningTest
#PBS -l nodes=4:ppn=32,mem=10gb

#---------------------------------------------------------------------------------
# Print some useful variables

echo "Job ID: bayesian_learning_test"
echo "Job User: arjung"
echo "Num Cores: 1"
start_time=$SECONDS

#---------------------------------------------------------------------------------
module load openmpi/1.10.7
module load python3/anaconda

source activate py396

#---------------------------------------------------------------------------------
cd quantmarketing/final/Code
python3 main.py

elapsed=$(( SECONDS - start_time ))
eval "echo Elapsed time: $(date -ud "@$elapsed" +'$((%s/3600/24)) days %H hr %M min %S sec')"

