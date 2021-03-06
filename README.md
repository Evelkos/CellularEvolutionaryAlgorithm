# CellularEvolutionaryAlgorithm
Cellular evolutionary algorithm implementation.
AMHE (Algorytmy metaheurystyczne) project.

![Alt Text](https://github.com/Evelkos/CellularEvolutionaryAlgorithm/blob/main/f10/cEA/cEA_f10_10x10_5000_evolution_2D.gif)
![Alt Text](https://github.com/Evelkos/CellularEvolutionaryAlgorithm/blob/main/f10/cEA/cEA_f10_10x10_5000_evolution_3D.gif)


# Installation
1. Install basic libraries
    ```
    sudo apt-get install python3.8 python-dev python3.8-dev \

        build-essential libssl-dev libffi-dev \

        libxml2-dev libxslt1-dev zlib1g-dev \

        python-pip
    ```

    Source: https://github.com/scrapy/scrapy/issues/2115

2. Create virtual environment

    `virtualenv .venv -p <path to the Python 3.8>`

3. Run virtual environment

    `source <path to the .venv>/bin/activate`

4. Install requirements with pip

    `pip install -r requirements.txt`

5. Deactivate environment

    `deactivate`

6. [Optional] Install `ffmpeg` 
   `ffmpeg` is used to save `.mp4` files with `matplotlib`. It is still possible to
   record evolution without `ffmpeg`, but only using `.gif` extension.

# Functionality
- train CellularEvolutionaryAlgorithm or EvolutionaryAlgorithm
- analyse min, max and mean fitness values from each iteration
- analyse fitness values across the entire population (in different iterations)
- record evolution in 2D or 3D

# Example
See `src/example.py` to see how to use `cellular_algorithm`

# CEC 2017 benchmark

All CEC 2017 functions used in this project come from  `tilleyd/cec2017-py`.

Source: https://github.com/tilleyd/cec2017-py
