#!/bin/bash
python demo1_ball_position/scripts/main.py &
python demo2_sawblade/scripts/main.py &
python demo3_resistors/scripts/main.py &
python demo4_classification/scripts/main.py &
python webserver/webserver.py
wait