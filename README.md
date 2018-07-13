# speedrun

## What? 

A no-strings-attached toolkit to help you deploy and manage your machine learning experiments. The idea is to equip you with the tools you need to have well-documented and reproducible experiments going, but without _getting in your way_.  

### Installation
On python 3.6+, install by: 

```bash
# Clone the repository
git clone https://github.com/nasimrahaman/speedrun.git
cd speedrun/
# Install
python setup.py install
```

## How? 

speedrun provides the base-class `BaseExperiment` for your experiments, in addition to a Tensorboard plug-in: `TensorboardMixin`. `BaseExperiment` contains handy tools for commandline argument & yaml configuration parsing and basic checkpointing, all of which you're free and welcome to override and adapt to your requirements. `TensorboardMixin` thinly wraps tensorboardX to get you god-tier logging right out of the box (but is fully optional, in-case you like your logging your way). 

Here's how it's meant to work. 

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

Now, run the file: 

```bash
mkdir experiments
python my_experiment.py experiments/BASIC-0 
> --config.my_cool_module.kwargs "{'a': 1, 'b': 2}" \
> --config.another_module.kwargs "{'c': 3, 'd': 4}" \
> --config.training.num_iterations 100000 \
> --config.training.checkpoint_every 10000 \
> --config.tensorboard.log_images_every 100 \
> --config.tensorboard.log_scalars_every 10
```

This will create a directory `experiments/BASIC-0` with multiple subdirectories. The configuration will be dumped in `experiments/BASIC-0/Configurations`, the tensorboard logs in `experiments/BASIC-0/Logs` and the checkpoints in `experiments/BASIC-0/Weights`. Of course, a fully valid option would be to create `BASIC-0/Configurations/train_config.yml` manually (you'll usually only need to do this once!) and populate it with an editor. 

Now say you want to try another set of kwargs for your cool module. All you need to do is: 
```bash
python my_experiment.py experiments/BASIC-1 --inherit experiments/BASIC-0 --config.my_cool_module.kwargs "{'a': 42, 'b': 84}"
```

This will _inherit_ the configuration from `BASIC-0`, but override the kwargs of your cool module. The resulting configuration will be dumped in `experiments/BASIC-1/Configurations/train_config.yml` for future experiments to inherit from. 

To know the exact difference between the two experiments, you can always: 

```bash
diff experiments/BASIC-0/Configurations/train_config.yml experiments/BASIC-1/Configurations/train_config.yml
```

When you're done with the first round of experiments, it's super easy to iterate on your ideas simply by inheriting from your `MyFirstExperiment`, perhaps in a different file: 

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

Of course, it's hard to know in advance what parts of the experiment would eventually need to be replaced - so you might need to refactor `MyFirstExperiment` and move bits of logic to their own methods, which you can then overload in `MySecondExperiment`.

## Why?
![shitcode](https://i.imgur.com/qG08mar.jpg)

