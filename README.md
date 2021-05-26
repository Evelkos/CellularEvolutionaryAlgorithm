# CellularEvolutionaryAlgorithm
Cellular evolutionary algorithm implementation.
AMHE (Algorytmy metaheurystyczne) project.


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


# CEC 2017 benchmark

All CEC 2017 functions used in this project come from  `tilleyd/cec2017-py`.

Source: https://github.com/tilleyd/cec2017-py
