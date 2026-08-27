#!/bin/bash
#PBS -l nodes=1:ppn=32
#PBS -q workq
#PBS -N cvh_test
#PBS -e job.err
#PBS -o job.out

# Move to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Load Anaconda module
module load anaconda3-2021

# Activate your custom environment
source activate quantum_env

# Safely detect allocated cores
if [ -n "$PBS_NUM_PPN" ]; then
    CORES=$PBS_NUM_PPN
elif [ -f "$PBS_NODEFILE" ]; then
    CORES=$(wc -l < "$PBS_NODEFILE")
else
    CORES=1
fi

# Ensure integer export without empty strings
export JOBLIB_CPU_COUNT=$CORES
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES

echo "Job started on $(hostname) at $(date)" > log.txt
echo "Allocated Cores: $CORES" >> log.txt
echo "----------------------------------------" >> log.txt

# Record timing
start_time=$(date +%s)

# Run your Python script
python cvh_spin_1_half_test.py >> log.txt 2>&1

end_time=$(date +%s)
wall_time=$((end_time - start_time))

echo "----------------------------------------" >> log.txt
echo "Job finished at $(date)" >> log.txt
echo "Total execution time: ${wall_time} seconds" >> log.txt
