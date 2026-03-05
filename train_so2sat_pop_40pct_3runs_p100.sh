#!/bin/bash
# So2Sat_POP 40% labels – 3 runs for tmux
#
# Run each command in a separate tmux pane or window (interactive, no -d).
# No background containers; you control sessions in tmux.

echo "=========================================="
echo "SimRegMatch: So2Sat_POP 40% – run in tmux"
echo "=========================================="
echo ""
echo "Start tmux and run ONE of these in each pane/window:"
echo ""
echo "  # Pane/Window 1 (GPU 0, seed 0):"
echo "  ./train_so2sat_pop_40pct_single.sh 0 0"
echo ""
echo "  # Pane/Window 2 (GPU 1, seed 42):"
echo "  ./train_so2sat_pop_40pct_single.sh 1 42"
echo ""
echo "  # Pane/Window 3 (GPU 2, seed 123):"
echo "  ./train_so2sat_pop_40pct_single.sh 2 123"
echo ""
echo "Example tmux workflow:"
echo "  tmux new -s simreg"
echo "  # Split or new windows, then in each:"
echo "  cd /work/ammar/sslrp/SimRegMatch"
echo "  ./train_so2sat_pop_40pct_single.sh 0 0   # pane 1"
echo "  ./train_so2sat_pop_40pct_single.sh 1 42  # pane 2"
echo "  ./train_so2sat_pop_40pct_single.sh 2 123 # pane 3"
echo ""
echo "Results: HDD/AgeDB_dir/results/so2sat_pop/SimRegMatch/experiment_*"
echo ""
