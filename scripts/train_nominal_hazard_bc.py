"""Train nominal-winner actions on hazard-visible observations.

Uses compare_fake_real_hazard's placement constraints. The BC targets are
nominal actions at the same physical states, reconstructed with real-hazard
lidar observations. Uniform weights w_t=1; optional reference KL regularization.
W&B logging defaults to online. Evaluation uses the standard SAC-Lag evaluator
on held-out nominal environments (not the selected training trajectories).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.distill_hazard_winners import main

if __name__ == '__main__':
    sys.argv.append('--nominal-relabel')
    if not any(a == '--wandb-mode' or a.startswith('--wandb-mode=') for a in sys.argv):
        sys.argv.extend(['--wandb-mode', 'online'])
    main()
