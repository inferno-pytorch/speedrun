import yaml
from speedrun.yaml_utils import ObjectLoader


input = """

case: !Case:va.lue
    a: 
        x: y
        y: x
    b: True
    
ref: !Ref va

value: b

va:
    lue:
        a
"""

data = yaml.load(input, Loader=ObjectLoader)
loaded = yaml.load(input, Loader=yaml.Loader)
dumped = yaml.dump(loaded, sort_keys=False)
print(dumped)
data = yaml.load(dumped, Loader=ObjectLoader)

print(data)
