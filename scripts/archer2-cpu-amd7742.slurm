#!/bin/bash

#SBATCH --job-name=pyDGEMM
#SBATCH --time=0:20:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128

#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --account=z19

module load cray-python/3.10.10

size=8192

total_tasks=$(( SLURM_JOB_NUM_NODES * SLURM_TASKS_PER_NODE * SLURM_CPUS_PER_TASK ))
timestamp=$(date '+%Y%m%d%H%M')
outfile="pydgemm_${size}_${SLURM_JOB_NUM_NODES}nodes_${total_tasks}procs_${timestamp}.res"

echo ""
echo "Running on ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_TASKS_PER_NODE} processes per node, $SLURM_CPUS_PER_TASK threads per process" > $outfile

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo "srun --hint=nomultithread --distribution=block:block python3 python-dgemm.py --nsize ${size}" >> $outfile

srun --hint=nomultithread --distribution=block:block python3 ../src/python-dgemm.py --nsize ${size} >> $outfile

