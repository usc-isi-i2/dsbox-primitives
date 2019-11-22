#!/bin/bash
pip install -e git+https://gitlab.com/datadrivendiscovery/d3m@feaf49da34568bbf37f82fa6ffeb127631020199#egg=d3m --progress-bar off
pip install -e git+https://gitlab.com/datadrivendiscovery/common-primitives.git@d9ee09a8838a222cead2a093d03c623603e175f9#egg=common_primitives --progress-bar off
pip install -e git+https://gitlab.com/datadrivendiscovery/sklearn-wrap@dist#egg=sklearn-wrap --progress-bar off
pip uninstall -y tensorflow-gpu
export LD_LIBRARY_PATH="$HOME/miniconda/envs/ta1-test-env/lib:$LD_LIBRARY_PATH"
pip install tensorflow==1.12.0
pip install -e . --progress-bar off
pip install -e git+https://github.com/brekelma/dsbox_corex@master#egg=702c626638163145b2f4f573ac200383fa6b7fd3 --progress-bar off
pip list
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5 --no-verbose
mv resnet50_weights_tf_dim_ordering_tf_kernels.h5 bdc6c9f787f9f51dffd50d895f86e469cc0eb8ba95fd61f0801b1a264acb4819
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 --no-verbose
mv vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 bfe5187d0a272bed55ba430631598124cff8e880b98d38c9e56c8d66032abdc1
ls -l