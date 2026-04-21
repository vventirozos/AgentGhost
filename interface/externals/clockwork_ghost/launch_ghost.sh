#!/bin/bash
xset s off
xset -dpms
xset s noblank

source /home/vasilis/gui_env/bin/activate
export QT_QPA_PLATFORM=xcb

python3 /home/vasilis/bin/client.py
