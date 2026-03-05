#!/bin/bash
# So2Sat_POP 40% labels – 3 runs for tmux (V100S/V100 GPUs)
#
# Run each command in a separate tmux pane or window (interactive, no -d).

echo "=========================================="
echo "SimRegMatch: So2Sat_POP 40% – V100S/V100 (tmux)"
echo "=========================================="
echo ""
echo "Run ONE of these in each tmux pane/window:"
echo ""
echo "  # Pane/Window 1 (GPU 4, seed 0):"
echo "  ./train_so2sat_pop_40pct_single.sh 4 0"
echo ""
echo "  # Pane/Window 2 (GPU 5, seed 42):"
echo "  ./train_so2sat_pop_40pct_single.sh 5 42"
echo ""
echo "  # Pane/Window 3 (GPU 7, seed 123):"
echo "  ./train_so2sat_pop_40pct_single.sh 7 123"
echo ""
echo "Example: tmux new -s simreg && cd $(pwd)"
echo "Results: HDD/AgeDB_dir/results/so2sat_pop/SimRegMatch/experiment_*"
echo ""
