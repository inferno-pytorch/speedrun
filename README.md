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
# Install tensorboard
pip install tensorboardX
# Install dill
pip install dill
# Install Weights and Biases
pip install wandb
```

## How? 

At the most basic level, speedrun provides the base-class `BaseExperiment` for your experiments to inherit from. This already enables you to read from configuration files and manages the experiment directories for you, so you have the choice of not worrying about file-paths and other nasty low-level details. 

But in addition, there are numerous fully optional power-ups that you can bolt on to your experiment class to make it what you need. These are called `Mixin`s, and the growing catalogue includes: 

- `WandBMixin` for out-of-the-box logging to [Weights and Biases](https://www.wandb.com/).
- `TensorboardMixin` to log runs locally to a tensorboard log-file (wraps [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)). 
- `WandBSweepMixin` and `SweepRunner` to configure and launch hyper-parameter sweeps with [Weights and Biases Sweeps](https://docs.wandb.com/sweeps). 
- `IOMixin` to supply utilities for e.g. writing text or images to file, progress bars, etc.
- `MatplotlibMixin` to convert matplotlib figures to images that you can then log to tensorboard with the `TensorboardMixin` or dump to file with the `IOMixin`. 
- `FirelightMixin` to interact with [Firelight](https://github.com/inferno-pytorch/firelight), a tool for visualizing high-dimensional embeddings. 
- `InfernoMixin` to interact with [Inferno](https://github.com/inferno-pytorch/inferno), a tool to abstract away the training loop for your pytorch models. 
- `WaiterMixin` to have your code wait for a running process to finish (and release resources). 

... and more underway. Check out [speedrun-springboard](https://github.com/inferno-pytorch/speedrun-springboard) for a prefabricated experimental set-up, or the small pseudo-example below: 

```python
from speedrun import BaseExperiment, TensorboardMixin


class MyFirstExperiment(BaseExperiment, TensorboardMixin):
    def __init__(self):
        super(MyFirstExperiment, self).__init__()
        # This is where the magic happens
        self.auto_setup()
        # Set up your experiment here
        ...
        self.my_cool_module = SomeModule(**self.get('my_cool_module/kwargs'))
        self.another_component = SomeComponent(**self.get('another_component/kwargs', default={}))
        # Say you have a component that gets messy and uses unpickleable objects. For checkpointing 
        # to still work, you'll need to tell the base experiment to not try pickle it. 
        self.ugly_messy_component = UglyMessyComponent()
        self.register_unpickleable('ugly_messy_component')
        ...
    
    def some_basic_logic(self, *args):
        # ...
        return self.bundle(result_1=..., result_2=..., result_3=...)
    
    def moar_logics(self):
        # ...
        # Uh oh, we need a global variable
        if 'one_time_computation_result' not in self.cache_keys:
            # Do the one time computation
            one_time_computation_result = self.some_basic_logic(self.step % 10)
            self.write_to_cache('one_time_computation_result', one_time_computation_result)
        else:
            one_time_computation_result = self.read_from_cache('one_time_computation_result')
        # ...
        return self.bundle(result_1=...)
    
    def run(self):
        # ...
        for iteration in range(self.get('training/num_iterations')):
            # training maybe? 
            basic_results = self.some_basic_logic()
            new_result = self.moar_logics(basic_results.result_1, basic_results.result_2)
            output_sample = ...
            loss = ...
            if self.log_scalars_now: 
                self.log_scalar('training/loss', loss)
            if self.log_images_now: 
                self.log_image('training/output_sample', output_sample)
            # force=False would checkpoint if the step count matches current iteration
            self.checkpoint(force=False)
            # This increments the step counter
            self.next_step()

if __name__=='__main__': 
    MyFirstExperimet().run()
```

Now, there are a few simple steps before we can run the first experiment of the project. The subsequent experiments are a breeze! 

First, we make a directory to store the _experiment templates_ (you can call the directory anything you like).
```bash
mkdir templates
```
Next, let's make the first experiment template. You can call the template anything you like, but we'll call it `BASIC-X`. 
```bash
mkdir -p templates/BASIC-X/Configurations
nano templates/BASIC-X/Configurations/train_config.yml
```

Now, we paste in the following configuration in `train_config.yml`. Note that the `.../Configurations/train_config.yml` structure is required for speedrun to find it. 

```yml
my_cool_module:
  kwargs: 
    a: 1
    b: 2
another_module:
  kwargs: 
    c: 3
    d: 4
training: 
  num_iterations: 100000
  checkpoint_every: 10000
tensorboard: 
  log_images_every: 100
  log_scalars_every: 10
```

Finally, we make a directory for our actual experiments (not the templates) to live in. You can call it anything you like, but we'll call it `experiments`: 
```bash
mkdir experiments
```

That's it, we're all set! To launch our first experiment, which we call `BASIC-0`, we could do:  
```bash
python my_experiment.py experiments/BASIC-0 --inherit templates/BASIC-X
```
This will create a directory `experiments/BASIC-0` with multiple subdirectories. The configuration will be dumped in `experiments/BASIC-0/Configurations`, the tensorboard logs in `experiments/BASIC-0/Logs` and the checkpoints in `experiments/BASIC-0/Weights`. 

So you fire up your first experiments but you think the keyword argument `a` of `my_cool_module` should instead be `42` and `d` of `another_module` should be `21`. All you need to do is: 
```bash
python my_experiment.py experiments/BASIC-1 --inherit experiments/BASIC-0 --config.my_cool_module.kwargs.a 42 --config.another_module.kwargs.d 21
```

This will _inherit_ the configuration from `BASIC-0`, but override `a` in kwargs of `my_cool_module` and `d` in kwargs of `another_module`. The resulting configuration will be dumped in `experiments/BASIC-1/Configurations/train_config.yml` for future experiments to inherit from! This way, you can iterate over your experiments and be confident that every run is self-contained and reproducible. To know the exact difference between the two experiments, you can always: 
```bash
diff experiments/BASIC-0/Configurations/train_config.yml experiments/BASIC-1/Configurations/train_config.yml
```

The tools might be nice, but it's not just just about that - organizing experiments in classes is a great way of reusing code, which in turn helps keep your experiments reproducible. Say when you're done with the first round of experiments, it's super easy to iterate on your ideas simply by inheriting from your `MyFirstExperiment`, perhaps in a different file: 

```python
from main import MyFirstExperiment

class MySecondExperiment(MyFirstExperiment):
    def moar_logics(self):
        # Your shiny new logics go in here
        # ...
        return self.bundle(result_1=...)

if __name__=='__main__':
    MySecondExperiment().run()
```

This way, when you fix a bug in `MyFirstExperiment.some_basic_logic`, it's automatically fixed in `MySecondExperiment` as well. Fine print: it's hard to know in advance what parts of the experiment would eventually need to be replaced - so you might need to refactor `MyFirstExperiment` and move bits of logic to their own methods, which you can then overload in `MySecondExperiment`. But more often than not, it's totally worth the effort. 

## Why?
![shitcode](https://i.imgur.com/qG08mar.jpg)

