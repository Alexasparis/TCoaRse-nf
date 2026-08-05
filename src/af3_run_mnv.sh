#!/bin/bash
#SBATCH --job-name=AF3                # Set the job name to "AF3"
#SBATCH --time=48:00:00               # Set the maximum wall time
#SBATCH --cpus-per-task=20            # Request 20 CPUs per task
#SBATCH --gres gpu:1                  # Request 1 GPU for the job
#SBATCH --ntasks=1                    # Request only 1 task
#SBATCH --output=logs/AF3_%j.out      # Standard output will be saved in logs/AF3_JOB_ID.out
#SBATCH --error=logs/AF3_%j.err       # Standard error will be saved in logs/AF3_JOB_ID.err
#SBATCH --account=bsc72               # Specify the account to charge for job resources
#SBATCH -D .                          # Set the working directory to the current directory
#SBATCH --qos=acc_bscls               # Request the "acc_bscls" quality of service

# Load necessary modules
module purge                           # Clean up any loaded modules
module load singularity cuda/12.6      # Load Singularity container and CUDA 12.6
module load alphafold/3.0.1            # Load AlphaFold 3.0.0 module

# Set the number of CPUs per task in the environment variable for use in the script
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

echo "Start $(date)"
echo "---------------"

# Define input and output directories
IN_FOLDER_WITH_JSONS=$1                    # Folder with input subfolders with JSONs
OUT_FOLDER=$2                              # Output directory
WEIGHTS=/gpfs/projects/bsc72/weights/AF3/  # Path to the pre-trained AlphaFold weights

# Check if the logs directory exists, create if necessary
mkdir -p ./logs

# CSV file for logging results
RESULTS_CSV=./logs/job_results.csv

# If the CSV file doesn't exist, create it with a header
if [ ! -f "$RESULTS_CSV" ]; then
    echo "subfolder,job_id,start,end,elapsed,status" > "$RESULTS_CSV"
fi

# Process each subfolder
for subfolder in $IN_FOLDER_WITH_JSONS/*; do
    START_TIME=$(date +%s)

    if bsc_alphafold "$subfolder" "$OUT_FOLDER" "$WEIGHTS"; then
        STATUS="OK"
    else
        # If error
        STATUS="ERROR"
        END_TIME=$(date +%s)
        ELAPSED_TIME=$((END_TIME - START_TIME))
        START_TIME_HR=$(date -d @$START_TIME +"%Y-%m-%d %H:%M:%S")
        END_TIME_HR=$(date -d @$END_TIME +"%Y-%m-%d %H:%M:%S")
        echo "$subfolder,$SLURM_JOB_ID,$START_TIME_HR,$END_TIME_HR,$ELAPSED_TIME,$STATUS" >> "$RESULTS_CSV"
        continue
    fi

    # If no errors
    END_TIME=$(date +%s)
    ELAPSED_TIME=$((END_TIME - START_TIME))
    START_TIME_HR=$(date -d @$START_TIME +"%Y-%m-%d %H:%M:%S")
    END_TIME_HR=$(date -d @$END_TIME +"%Y-%m-%d %H:%M:%S")
    echo "$subfolder,$SLURM_JOB_ID,$START_TIME_HR,$END_TIME_HR,$ELAPSED_TIME,$STATUS" >> "$RESULTS_CSV"
done

echo "End $(date)"
echo "--------------"