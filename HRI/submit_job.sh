#!/bin/bash
#PBS -l nodes=1:ppn=32
#PBS -q workq
#PBS -N cvh_test
#PBS -e job_${PBS_JOBID}.err
#PBS -o job_${PBS_JOBID}.out

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

# Clean job ID (optional: removes the hostname suffix, turning '12345.master1' into '12345')
JOB_ID=$(echo $PBS_JOBID | cut -d'.' -f1)

# Log file with Job ID prefix (e.g., 12345_log.txt)
LOGFILE="${JOB_ID}_log.txt"

echo "Job started on $(hostname) at $(date)" > "$LOGFILE"
echo "Allocated Cores: $CORES" >> "$LOGFILE"
echo "----------------------------------------" >> "$LOGFILE"

# Record timing
start_time=$(date +%s)

# Run your Python script
python cvh_spin_1_half_test.py >> "$LOGFILE" 2>&1

end_time=$(date +%s)
wall_time=$((end_time - start_time))

echo "----------------------------------------" >> "$LOGFILE"
echo "Job finished at $(date)" >> "$LOGFILE"
echo "Total execution time: ${wall_time} seconds" >> "$LOGFILE"
