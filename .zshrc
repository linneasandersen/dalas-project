
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/Users/linneaandersen/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/Users/linneaandersen/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/Users/linneaandersen/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/Users/linneaandersen/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<





[ -f "/Users/linneaandersen/.ghcup/env" ] && . "/Users/linneaandersen/.ghcup/env" # ghcup-env
# Pyenv setup
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


