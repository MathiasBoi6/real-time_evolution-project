# rt-nmmo
NMMO adjusted for real-time evolution learning.
Based on: https://github.com/NeuralMMO


# Setup
install nmmo=2.0.3
copy the file terrain.py from this folder and use it to replace the terrain.py file in the installed neuralMMO folder. This changes the map such that the center of the map is not just covered by impassable stone tiles.

Currently, the relevant file to run is newSpawn_main.py.

Upload resulting JSON for visualization of the run:
https://neuralmmo.github.io/client/


# Todo:
- find good metrics to track
- figure out a configuration that minimizes the risk of constant extinction
- which parts of the neuralMMO mechanisms should be used (communication system, market place, etc)?

