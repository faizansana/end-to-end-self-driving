# End to End Learning of Self Driving Vehicle in Urban Environments

This repository contains the code for the project submitted for CS686. The architecture of the neural network is based off the architecture created by NVIDIA in [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf) famously known as PilotNet.

## Running the Code

### Disclaimer
Although this code was tested on Windows 11 Version 22H2, this should also work on Linux as well as MacOS. In the case of MacOS, you will most likely have to use an external CARLA server.
### Prerequisites

1. [Anaconda](https://www.anaconda.com/products/distribution) / [Miniconda](https://conda.io/projects/conda/en/stable/user-guide/install/download.html)

2. [CARLA Simulator](https://carla.org/) version [0.9.13](https://github.com/carla-simulator/carla/releases/tag/0.9.13)


### Steps

1. Clone the repository

    ```
    git clone https://github.com/faizansana/cs686-project.git
    ```
2. From within the working directory, create the conda environment.

    ```
    conda create -f environment.yml -n some_name_that_you_want
    ```
3. Depending on whether you want to collect data, train or test an already built model in simulation, you will use the following scripts as shown below

## Usage

### Generating Training/Test Data

Make sure the CARLA server has been started before running this script!

Run the `collect_data.py` file. This takes one positional argument and the usage can also be seen by using

```
python collect_data.py -h
```

To run `collect_data.py` for 2 minutes with a local CARLA server exposed on port 2000

```
python collect_data.py 2
```

This will then run data collection for 2 minutes and be stored in a folder called `recordings/{timestamp}/`.

The images are stored with the following naming convention:

```
[time_in_seconds, steering_angle, throttle, brake_pressure]
```

### Training the Model

You do not need the CARLA server for this case.

Run the `model.py` file. This only has optional arguments.

To generate a model with the name `first_model_epoch_100` with 100 epochs

```
python model.py --model-name "first_model_epoch_100" --epochs 100
```

### Testing the Model in Live Simulation

This section requires a CARLA server to be running.

Run the `run_model_in_carla.py` file which requires one positional argument.

In order to run simulation on a model stored in the path `./models/first_model_epoch_100`

```
python run_model_in_carla.py ./models/first_model_epoch_100
```

