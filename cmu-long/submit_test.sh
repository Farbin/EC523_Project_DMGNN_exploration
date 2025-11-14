#!/bin/bash -l

#$ -P epic-iarpa      # Specify the SCC project name you want to use
#$ -l h_rt=1:00:00   # Specify the hard time limit for the job
#$ -N test_cmu_long   # Give job a name
#$ -o test_cmu_long.txt
#$ -e test_cmu_long_error.txt
#$ -l gpus=1
#$ -l gpu_type=H200


export XDG_CACHE_HOME=/projectnb/epic-iarpa/ffayza/.cache
source /projectnb/epic-iarpa/ffayza/venvs/triosim-env/bin/activate

cd /projectnb/ec523bn/students/ffayza/project/DMGNN/cmu-long/
python3 main.py prediction -c ../config/CMU/long/test.yaml
