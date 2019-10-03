"""
speedrun - A toolkit for quick-and-clean machine learning experiments with Pytorch and beyond.
"""

import setuptools


setuptools.setup(
    name="speedrun",
    author="Nasim Rahaman",
    author_email="nasim.rahaman@iwr.uni-heidelberg.de",
    license='GPL-v3',
    description="Toolkit for machine learning experiment management.",
    version="0.1",
    install_requires=['pyyaml==4.2b4'],
    packages=setuptools.find_packages(),
)
