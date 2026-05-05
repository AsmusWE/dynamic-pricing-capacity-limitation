#!/bin/sh
#================ LSF OPTIONS =================

# ── Edit the array range to match the number of configs below ──
#BSUB -J MARL_sweep[1-8]

#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 03:00
#BSUB -u aswin@dtu.dk
#BSUB -o Output_%J_%I.out
#BSUB -e Output_%J_%I.err

#=============== HYPERPARAMETER CONFIGS ================
# Add one entry per config. Keep all arrays the same length
# and update the [1-N] range in #BSUB -J above to match.
#
# Hydra overrides: append any marl_main.py config key as key=value.
# Leave wandb_run_name empty to auto-generate from hyperparameters.

RUN_NAMES=(
    "default"
    "entropy_0.5"
    "hidden_128"
    "hidden_512"
    "tau_0.05"
    "tau_0.001"
    "lr_1e-5"
    "lr_5e-6"
)

CONFIGS=(
    # 1 - default (no overrides beyond what config_dev sets)
    "episodes=8000"
    # 2 - lower target entropy multiplier
    "episodes=8000 target_entropy_multiplier=0.5"
    # 3 - smaller network
    "episodes=8000 hidden_dim=128"
    # 4 - larger network
    "episodes=8000 hidden_dim=512"
    # 5 - faster target network updates
    "episodes=8000 tau=0.05"
    # 6 - slower target network updates
    "episodes=8000 tau=0.001"
    # 7 - low learning rates
    "episodes=8000 actor_lr=1e-5 critic_lr=1e-5 alpha_lr=1e-5"
    # 8 - very low learning rates
    "episodes=8000 actor_lr=5e-6 critic_lr=5e-6 alpha_lr=5e-6"
)

#=======================================================

#=============== ENVIRONMENT ==================

module purge
module load python3/3.13.11

cd ~/dynamic-pricing-capacity-limitation || exit 1
source venv/bin/activate

echo "Job array index: $LSB_JOBINDEX"
echo "Python: $(python --version)"
python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

#================= RUN CODE ===================

IDX=$((LSB_JOBINDEX - 1))   # bash arrays are 0-indexed; LSB_JOBINDEX starts at 1
RUN_NAME="${RUN_NAMES[$IDX]}"
OVERRIDES="${CONFIGS[$IDX]}"

echo "Run name : $RUN_NAME"
echo "Overrides: $OVERRIDES"

cd $HOME/dynamic-pricing-capacity-limitation
source venv/bin/activate

python marl_main.py \
    --config-name config_dev \
    use_wandb=true \
    wandb_run_name="$RUN_NAME" \
    plot_path="Figures/${RUN_NAME}_returns.png" \
    benchmark_actions_path="Figures/${RUN_NAME}_benchmark_actions.png" \
    $OVERRIDES
