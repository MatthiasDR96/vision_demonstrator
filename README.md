# Vision Demonstrator
Software for the vision demonstrator of KU Leuven Bruges

## Installation

### Install IDS peak

* Install IDS peak 2.4 from https://en.ids-imaging.com/download-details/AB12780.html
* Go to 'C:\Program Files\IDS\ids_peak\generic_sdk\api\binding\python\wheel\x86_64' in command prompt
* Do 'pip install ids_peak-1.6.0.0-cp38-cp38-win_amd64.whl'
* C:\Program Files\IDS\ids_peak\generic_sdk\ipl\binding\python\wheel\x86_64
* Do 'pip install ids_peak_ipl-1.7.0.0-cp38-cp38-win_amd64.whl'
* C:\Program Files\IDS\ids_peak\generic_sdk\afl\binding\python\wheel\x86_64
* Do 'pip install ids_peak_afl-1.1.0.0-cp38-cp38-win_amd64.whl'

### Connection to Omron Vision Controller

Set netwerk adapter to IP 10.5.5.110 with mask 255.255.255.0

## Run demo

To run the demo, each of the main scripts can be executed using the bash scripts in 'bash'. All scripts will publish their output to a MQTT server that is subscribed by the webserver. The result of the webserver is visible via 'localhost:5000'.

