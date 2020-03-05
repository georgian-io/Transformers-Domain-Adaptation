#!/bin/sh
SESSION='dpt'

# Find or create a tmux session
if tmux has-session -t $SESSION 2> /dev/null; then
    tmux attach-session -t $SESSION -d
else
    tmux new-session -s $SESSION -d
fi

# First window for general stuff
tmux send-keys "sudo yum install zsh -y" C-m
tmux send-keys "exec zsh" C-m
tmux send-keys "source activate pytorch_p36" C-m
tmux send-keys "pip install -U pip" C-m
tmux send-keys "pip install -r requirements.txt" C-m

# Install apex
tmux send-keys "git clone https://github.com/NVIDIA/apex" C-m
tmux send-keys "cd apex" C-m
tmux send-keys "pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./" C-m
tmux send-keys "cd ~/SageMaker/NLP-Domain-Adaptation" C-m

# Second window for htop
tmux split-window -v
tmux send-keys "sudo yum install htop -y" C-m
tmux send-keys "htop" C-m

# Third window for GPU usage
tmux split-window -h
tmux send-keys "watch -n 1 nvidia-smi" C-m
