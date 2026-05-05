#!/bin/sh
#================ LSF OPTIONS =================

# ── Edit the array range to match the number of configs below ──
#BSUB -J MARL_sweep[1-8]

#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4B]"
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
    "target_entropy_multiplier=0.5"
    "hidden_dim=128"
    "hidden_dim=512"
    "tau=0.05"
    "tau=0.001"
    "actor_lr=1e-5 critic_lr=1e-5 alpha_lr=1e-5"
    "actor_lr=5e-6 critic_lr=5e-6 alpha_lr=5e-6"
)

CONFIGS=(
    # 1 - default
    "plot_path=Figures/default_returns.png benchmark_actions_path=Figures/default_benchmark_actions.png"
    # 2 - target_entropy_multiplier=0.5
    "target_entropy_multiplier=0.5 plot_path=Figures/target_entropy_0.5_returns.png benchmark_actions_path=Figures/target_entropy_0.5_benchmark_actions.png"
    # 3 - hidden_dim=128
    "hidden_dim=128 plot_path=Figures/hidden_dim_128_returns.png benchmark_actions_path=Figures/hidden_dim_128_benchmark_actions.png"
    # 4 - hidden_dim=512
    "hidden_dim=512 plot_path=Figures/hidden_dim_512_returns.png benchmark_actions_path=Figures/hidden_dim_512_benchmark_actions.png"
    # 5 - tau=0.05
    "tau=0.05 plot_path=Figures/tau_0.05_returns.png benchmark_actions_path=Figures/tau_0.05_benchmark_actions.png"
    # 6 - tau=0.001
    "tau=0.001 plot_path=Figures/tau_0.001_returns.png benchmark_actions_path=Figures/tau_0.001_benchmark_actions.png"
    # 7 - actor_lr=1e-5 critic_lr=1e-5 alpha_lr=1e-5
    "actor_lr=1e-5 critic_lr=1e-5 alpha_lr=1e-5 plot_path=Figures/lr_1e-5_returns.png benchmark_actions_path=Figures/lr_1e-5_benchmark_actions.png"
    # 8 - actor_lr=5e-6 critic_lr=5e-6 alpha_lr=5e-6
    "actor_lr=5e-6 critic_lr=5e-6 alpha_lr=5e-6 plot_path=Figures/lr_5e-6_returns.png benchmark_actions_path=Figures/lr_5e-6_benchmark_actions.png"
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

python marl_main.py \
    --config-name config_dev \
    use_wandb=true \
    wandb_run_name="$RUN_NAME" \
    plot_path="Figures/${RUN_NAME}_returns.png" \
    benchmark_actions_path="Figures/${RUN_NAME}_benchmark_actions.png" \
    $OVERRIDES
