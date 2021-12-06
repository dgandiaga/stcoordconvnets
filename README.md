# Spatial Transforms and CoordConv integration

This is a dockerized implementation of [Spatial Transform Netrworks](https://arxiv.org/abs/1506.02025) with [CoordConv layers](https://arxiv.org/abs/1807.03247) that allows to train some models and save and interact with the results through Jupyter Notebook.

## Features

- Train different architectures from a scratch
- Test them on [MNIST](http://yann.lecun.com/exdb/mnist/) and [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- Save, load and compare results
- Everything is dockerized in order to allow a fast deployment


## Usage

This repository has been containerized through [docker compose](https://docs.docker.com/compose/) (version 1.28.2)

### Jupyter-lab
Runs the dockerized jupyter-lab server.

```sh
docker-compose up  jupyter
```

It copies the source code and the results and models to the docker container in order to analyze them through jupyter. It mounts a volume on folder *jupyter* and maps it to *work/jupyter* in the container so you can persist your modifications in your host if you decide to make some changes in the analysis notebook. It may require you to grant permissions so the docker container can write in the original folder.

The analytics notebook route in the container is **work/jupyter/result_analysis.ipynb**
There you'll see:
* Accuracy metrics of the different models averaged through many runs with their standard deviations
* Visualizations of the images after the Spatial Transform layer for the models that have them
* Confusion Matrixes
* Examples of the misclassified items and insights on them

### Train models
The training loop is also dockerized. You just need to run:

```sh
docker-compose build stnet
docker-compose run stnet
```

It mounts and maps volumes in folders *results*, *models* and *datasets* so you can persist your trained models and data in your host's route.

In order to change the parameters you'll have to edit the service configuration in docker-compose.yml file:

```sh
services:
  stnet:
    build:
      context: .
      dockerfile: Dockerfile.train
    container_name: stnet
    tty: true
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./results/:/code/results/
      - ./models/:/code/models/
      - ./datasets/:/code/datasets/
    command: 'python train.py --dataset fashion-mnist --model stcoordconv --epochs 30'
    runtime: nvidia
```

There you can see the command with the arguments. Allowed values are:

* Dataset:
    * mnist
    * fashion-mnist

* Model:
    * convnet: Classical CNN 
    * stnet: vanilla STnet as in https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
    * coordconv: CNN with coordconv layers instead of classical conv layers as in https://github.com/walsvid/CoordConv adapted to PyTorch 1.x
    * stcoordconv: Combination of STN and coordconv implementations

If your system lacks of GPU capabilities or you have not enabled the [nvidia-dockercompose integration](https://docs.docker.com/compose/gpu-support/) properly you may have to edit the docker-compose.yml and remove both **runtime: nvidia** refferences in order to be able to launch the project. 

## Results

Here we can see a comparison between 10 runs of different architectures over fashion-MNIST dataset:
![image](https://user-images.githubusercontent.com/26325749/144834352-2ed3e471-aaf2-4d77-9d14-712db490dcf6.png)

In the jupyter notebook there are also other metrics and visualizations implemented. For example, you can check confusion matrixes over the test set:
![image](https://user-images.githubusercontent.com/26325749/144834535-54267f7c-fd9b-4554-ac20-ab94705b5d82.png)

And also see how the spatial transform layers transform the imput images in the models that include them:
![image](https://user-images.githubusercontent.com/26325749/144834651-d9e00112-cfb4-47a2-a337-2c50e6838806.png)

