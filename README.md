 # HMS: A Hierarchical Solver with Dependency-Enhanced Understanding for Math Word Problem
Source code for paper *HMS: A Hierarchical Solver with Dependency-Enhanced Understanding for Math Word Problem*.

 ## Dependencies
- python >= 3.6

- stanfordcorenlp
- torch

 ## Usage
Preprocess dataset
```bash
python3 src/dataprocess/math23k.py
```
Train and test model
```bash
python3 src/main.py
```
For running arguments, please refer to [src/config.py](src/config.py).