# Real-Time-NMMO

Neural MMO adjusted for real-time evolution learning.
Based on: https://github.com/NeuralMMO

# Setup
Install nmmo=2.0.3. Alternatively, download all dependencies in a conda environment with the `rt_evo.yml` file using 
```
conda env create -f environment.yml
```

Copy the file terrain.py from this folder and use it to replace the terrain.py file in the installed NeuralMMO folder. This changes the map such that the center of the map is not just covered by impassable stone tiles.

# Experiments and files

Original Real-time NMMO remade with better memory usage [RemadeNewspawns.py](./RemadeNewspawns.py), allows agents to produce offspring, simulating natural selection while evolving the agents. The original version of this code would never deallocate data, but this has been fixed in our code.

Added crossover [DiverseBirths.py](DiverseBirths.py), where crossover is added when new agents are created to help search the parameter space of the agents neural networks.

Double agents and 1.5 size map [DivBirthDouble.py](DivBirthDouble.py).

No offspring, instead setting a step limit [DiverseSearchHalfMC.py](DiverseSearchHalfMC.py). If step limit is reached, it is doubled for next era.

No offspring, instead setting a step limit [10MC.py](10MC.py). If step limit is reached, it increased by 1.1 for the next era.

No offspring, stopping at top 10% agents [RandomVision.py](./RandomVision.py). For this experiment, all agents have a copy of the same convolutional network, and mutation does not affect it. In this way, the agents have a randomly initialized Convolutional network.


# Rendering

Upload resulting JSON for visualization of the run:
https://neuralmmo.github.io/client/

# Todo:
- find good metrics to track
- figure out a configuration that minimizes the risk of constant extinction
- which parts of the neuralMMO mechanisms should be used (communication system, market place, etc)?

