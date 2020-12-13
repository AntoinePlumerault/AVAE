Setup
=====

To set up the environment you can use the Dockerfile if you have nvidia-docker installed on your machine.

Alternatively, you can create a Conda environment with the required packages by running the following commands in a terminal:

```bash
conda create -n icml_avae
conda activate icml_avae

conda install matplotlib
conda install tensorflow-gpu
pip install tensorflow-datasets
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install opencv
conda install scikit-image
conda install ipython
pip install tensorflow-hub
conda install Pillow=6.2.1
```

CelebA dataset
--------------

There is currently an issue with the tensorflow-datasets version of the CelebA dataset. To use this dataset you must follow the instructions in the `src/datasets/celeba` file.

Usage
=====

To run the experiments from the article you can use `scipts/example_script.sh`. For this, go to `src` then run it:

```bash
cd src/
../scripts/example_script.sh
```

Logging message, images, checkpoints and saved models are stored in the `outputs` directory. 

Alternatively you can directly run the `main.py` file with your choice of flags by following the example of `example_script.sh`.