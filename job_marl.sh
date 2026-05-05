#!/bin/bash
#================ LSF OPTIONS =================

# Job name
#BSUB -J marl_sac_a100

# Queue for A100 GPUs
#BSUB -q gpua100

# Number of CPU cores
#BSUB -n 4

# Request 1 GPU
#BSUB -gpu "num=1"

# Memory per core
#BSUB -R "rusage[mem=5GB]"

# Wall clock limit (hh:mm)
#BSUB -W 12:00

# Output and error files
#BSUB -o marl_gpu_%J.out
#BSUB -e marl_gpu_%J.err

#BSUB -u aswin@dtu.dk
#BSUB -R "span[hosts=1]"
#=============== ENVIRONMENT ==================

module purge

# Load Python (must match how your venv was created)
module load python3/3.13.11

# Go to project directory
cd ~/dynamic-pricing-capacity-limitation || exit 1

# Activate virtual environment
source venv/bin/activate

# Optional diagnostics (recommended)
echo "Python:"
python --version
echo "CUDA:"
python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

#================= RUN CODE ===================

python marl_main.py --wandb --episodes 10000 --warmup-steps 3000 --buffer-size 10000 --batch-size 1024 --updates-per-step 2