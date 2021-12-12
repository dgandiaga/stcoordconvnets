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
docker-compose build train
docker-compose run train
```

It mounts and maps volumes in folders *results*, *models* and *datasets* so you can persist your trained models and data in your host's route.

In order to change the parameters you'll have to edit the service configuration in docker-compose.yml file:

```sh
  train:
    build:
      context: .
      dockerfile: docker/Dockerfile.train
    container_name: train
    tty: true
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./results/:/code/results/
      - ./models/:/code/models/
      - ./datasets/:/code/datasets/
    command: 'python train.py --dataset fashion-mnist --model stnet --epochs 20'
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

This visualizations are extracted from the Jupyter notebook provided in jupyter service. 
Here we can see a comparison between different architectures over fashion-MNIST dataset, averaged over many runs:
![image](https://user-images.githubusercontent.com/26325749/144834352-2ed3e471-aaf2-4d77-9d14-712db490dcf6.png)

In the jupyter notebook there are also other metrics and visualizations implemented. For example, you can check confusion matrixes over the test set:
![image](https://user-images.githubusercontent.com/26325749/144834535-54267f7c-fd9b-4554-ac20-ab94705b5d82.png)

And also see how the spatial transform layers transform the imput images in the models that include them:
![image](https://user-images.githubusercontent.com/26325749/144834651-d9e00112-cfb4-47a2-a337-2c50e6838806.png)

## Bird Classification Dataset
I have added implementations for the bird classification dataset [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) they use in the [Spatial Transform Netrworks](https://arxiv.org/abs/1506.02025) paper:
![image](https://user-images.githubusercontent.com/26325749/145714926-c1d5aff5-5392-4539-aa0e-8fd9c07f2e8f.png)

I've tried to expand the implementation in https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html adding more complexity to the localization layer, but I've not gotten the expected results. Since the authors didn't release their code I have adapted this other implementation [STN with ResNext50 as backbone](https://github.com/hyperfraise/Pytorch-StNet) removing the use of initialized weights, since the goal of this project is comparing the results of the architectures and the convergence speed instead of training a state-of-the-art model, and updating the input structure which was designed for video instead of images. You can test them by choosing **birds** as dataset and **resnext** or **stresnext** as model.

The results don't seem to match the resnext baseline:
insdfsdfa

My conclusion is that **STNs** are an interesting technology but their implementation over different state-of-the-art challenges is not as straight forward as it may seem, and the literature about them is still in an early stage. They seem to require extensive running in the architecture in order to adapt them to a complex problem, and I'd choose them as an alternative for improving an already developed model, but I wouldn't start from here if the goal is fast-prototyping a new use case. A problem that they already mentioned in the paper is that they are **prone to overfitting**. My experiments corroborate that since the **training error** was **smaller** than in the resnext baseline but the **validation error** was **higher**.
Regarding my experiments they also **increase training time up to ~140%**, so you should consider that if you have time or infrastructure restrains. This seems worth in MNIST and fashion-MNIST since STN-based models go several epochs ahead of non-STN models, but it may not be justified in other contexts.
