export PATH="/opt/homebrew/opt/python@3.10/libexec/bin:$PATH"

export PATH=$PATH:/Users/nisumlimbu/.spicetify

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

export PATH="/Users/nisumlimbu/.config/herd-lite/bin:$PATH"
export PHP_INI_SCAN_DIR="/Users/nisumlimbu/.config/herd-lite/bin:$PHP_INI_SCAN_DIR"
export PATH="$PATH:$HOME/Library/Python/3.12/bin"
. "$HOME/.local/bin/env"
export PATH="$PATH:$HOME/Library/Python/3.12/bin"

