# Overcooked
## Contents of this repository
This repository reuses some of the code provided by 
[Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai)
in order to provide a multi-agent simulation scenario with an interface similar to that of OpenAi Gym. 
This scenario was used for the experiments of a MSc Thesis in Reinforcement Learning and Ad Hoc Teamwork, 
whose code is also included in the repository.

The code is organized as follows:
 - The *overcooked_ai_py* folder contains the code from 
[Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) used in this project. 
 - The *environment* contains the implementation of the environment in the *overcooked.py* file. 
This environment has a set of possible layouts. The one used in our experiments and for which the 
teammates were implemented is called "simple_kitchen".
 - The *agents* folder contains the implementation of the agents used. For someone only interested 
 in using the environment and not trying to replicate our experiments, only the *teammates* subfolder is
 relevant. It contains a set of handcoded agents that move and act in different ways, some more effective
 than other in the overcooked environment. They were only programmed, considering the "simple_kitchen"
 layout though, so they may not work as expected in different layouts. The remaining subfolders contain
 the agents used in the experiments of our work.
 - The *run_files* folder contains the implementation of our experiments and *data* the respective results.
 - The file *demo.py* is a simple demonstration to show how the environment works and can be used.
 
## Installation
We recommend the creation of a new virtual environment for this project.
In that new envioronment, install the [YAAF library](https://github.com/jmribeiro/yaaf) and then the 
[DQN library](https://github.com/ilvieira/deep-q-networks) used in this project. 
Follow the installation instructions from the respective repositories. After that, you can 
install the requirements of this repository and it will be functional. If you only want to use the 
environment, you can delete the folders above mentioned that are only related to the experiments for 
which it was used.
 

