#!/bin/sh
SESSION='dpt'

# Find or create a tmux session
if tmux has-session -t $SESSION 2> /dev/null; then
    tmux attach-session -t $SESSION -d
else
    tmux new-session -s $SESSION -d
fi

# First window for general stuff
tmux send-keys "source activate pytorch_p36" C-m
tmux send-keys "./run_pipeline.sh" C-m

# Second window for htop
tmux split-window -v
tmux send-keys "htop" C-m

# Third window for GPU usage
tmux split-window -h
tmux send-keys "watch -n 1 nvidia-smi" C-m
