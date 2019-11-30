# speedrun

## What? 

A no-strings-attached toolkit to help you deploy and manage your machine learning experiments. The idea is to equip you with the tools you need to have well-documented and reproducible experiments going, but without _getting in your way_. Think of it as a swiss army knife for dealing with the code-hell research projects typically tend to evolve to. 

### Installation
On python 3.6+:

```bash
# Clone the repository
git clone https://github.com/nasimrahaman/speedrun.git
cd speedrun/
# To embark on an adventure, uncomment the following line:
# git checkout dev
# Install
python setup.py install
```

Optionally, 

```bash
# Install tensorboardX
pip install tensorboardX
# Install dill
pip install dill
```

## How? 

At the most basic level, speedrun provides the base-class `BaseExperiment` for your experiments to inherit from. This already enables you to read from configuration files and manages the experiment directories for you, so you have the choice of not worrying about file-paths and other nasty low-level details. 

But in addition, there are numerous fully optional power-ups that you can bolt on to your experiment class to make it what you need. These are called `Mixin`s, and the growing catalogue includes: 

- `TensorboardMixin` to get you god-tier logging right of the box (wraps [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)). 
- `IOMixin` to supply utilities for e.g. writing text or images to file, progress bars, etc.
- `MatplotlibMixin` to convert matplotlib figures to images that you can then log to tensorboard with the `TensorboardMixin` or dump to file with the `IOMixin`. 
- `FirelightMixin` to interact with [Firelight](https://github.com/inferno-pytorch/firelight), a tool for visualizing high-dimensional embeddings. 
- `InfernoMixin` to interact with [Inferno](https://github.com/inferno-pytorch/inferno), a tool to abstract away the training loop for your pytorch models. 
- `WaiterMixin` to have your code wait for a running process to finish (and release resources). 

... and many more underway. Check out [speedrun-springboard](https://github.com/inferno-pytorch/speedrun-springboard) for a comprehensive example with PyTorch. 

### Overview

#### Setup
For starters, your experiments must be a class that inherits from `speedrun`'s `BaseExperiment`:

```python
from speedrun import BaseExperiment

class MyExperiment(BaseExperiment): 
    def __init__(self):
        super(MyFirstExperiment, self).__init__()
        # This is where the magic happens
        self.auto_setup()
        # Set up your experiment here; for example:
        self.model = ...
        self.data_loader = ...
    
    # This can be your "main" function, one that does the training. 
    def train(self): 
        # This is where you train your model
        for data in self.data_loader: 
            # your training logic goes in here. 
```

All you need to run your experiment is a python script (say `my_experiment.py`) with: 
```python
MyExperiment().run()
```
You can also append the this to the script where you have defined your experiment class, perhaps under a `if __name__ == '__main__'` block. You can run the script as following: 

```bash
mkdir experiments
python my_experiment.py experiments/MY-EXPERIMENT-0
```

#### Reading from Configuration and Arguments

When writing your experiment code, you can read from the configuration file via the `self.get` method, or from the commandline arguments via the `self.get_arg` method:

```python
class MyExperiment(BaseExperiment): 
    def __init__(self):
        super(MyFirstExperiment, self).__init__()
        self.auto_setup()
        self.model = MyModel(**self.get('model/kwargs', {}))
```

## Why?
![shitcode](https://i.imgur.com/qG08mar.jpg)

