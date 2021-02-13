AVAE: Adversarial Variational Auto Encoder
==========================================

A. Plumerault, H. Le Borgne, C. Hudelot (2021) AVAE: Adversarial Variational Auto Encoder, International Conference on Pattern Recognition (ICPR’20), Milan, Italy, 10-15 january 2021

If you use this code for your research, please site our paper:

```
@inproceedings{plumerault20icpr,
  title = {AVAE: Adversarial Variational Auto Encoder},
  author={Antoine Plumerault and Herv{\'e} {Le Borgne} and C{\'e}line Hudelot},
  booktitle={International Conference on Pattern Recognition},
  year={2020},
}
```

Setup
=====

To set up the environment you can use the Dockerfile if you have nvidia-docker installed on your machine.

Alternatively, you can create a python 3.7 Conda environment  with the required packages by running the following commands in a terminal:

```bash
conda create -n icpr_avae
conda activate icpr_avae

conda install matplotlib
conda install tensorflow-gpu
pip install tensorflow-datasets
# use the cudatoolkit that fit your tf.__version__
# see https://www.tensorflow.org/install/source#gpu
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
#conda install opencv
conda install -c conda-forge opencv
conda install scikit-image
###conda install ipython
pip install tensorflow-hub
```

You may also need to add a 'trick' to use the ptxas/xla
```
cd ${CONDA_DIST}/envs/icpr_avae/lib
mkdir -p nvvm/libdevice && cd nvvm/libdevice
ln -s ../../libdevice.10.bc .
```

Tested with:
- Python 3.8.5


CelebA dataset
--------------

There is currently an issue with the tensorflow-datasets version of the CelebA dataset. To use this dataset you must follow the instructions in the `src/datasets/celeba.py` file to create the TFrecords.

Usage
=====

To run the experiments from the article you can use `scipts/example_script.sh`. For this, go to `src` then run it:

```bash
cd src/
../scripts/example_script.sh
```

Logging message, images, checkpoints and saved models are stored in the `outputs` directory. 

Alternatively you can directly run the `main.py` file with your choice of flags by following the example of `example_script.sh`.

On your own dataset
--------------------
You can look at src/datasets/celeba.py to get an example
- create TFrecords with the main()
- create a dataset manager with the load()
- add the dataset to src/main.py. For axample for dataset FOO:
```
if FLAGS.dataset == 'foo': from datasets import foo as dataset
```
