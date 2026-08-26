#!/bin/bash
#PBS -l nodes=1:ppn=16
#PBS -q workq
#PBS -N joblib_python_job
#PBS -e job.err
#PBS -o job.out

# Move to the directory where the job was submitted
cd $PBS_O_WORKDIR

# Load your Python environment (adjust module name or activate your conda env)
# module load python/3.10
# source activate my_env

# Pass the allocated CPU count to Joblib / OpenMP threads
export OMP_NUM_THREADS=$PBS_NUM_PPN
export JOBLIB_CPU_COUNT=$PBS_NUM_PPN

echo "Job started on $(hostname) at $(date)" > log.txt
echo "Allocated Cores: $PBS_NUM_PPN" >> log.txt
echo "----------------------------------------" >> log.txt

# Record timing
start_time=$(date +%s)

# Run your Python script
python main.py >> log.txt 2>&1

end_time=$(date +%s)
wall_time=$((end_time - start_time))

echo "----------------------------------------" >> log.txt
echo "Job finished at $(date)" >> log.txt
echo "Total execution time: ${wall_time} seconds" >> log.txt