#!/bin/sh

isi=/opt/kyao-repo/dsbox/dsbox_primitives_repo/v2018.1.26/ISI
version=0.1.3
for suffix in dsbox.MultiTableFeaturization dsbox.Vgg16ImageFeature dsbox.ResNet50ImageFeature dsbox.RandomProjectionTimeSeriesFeaturization
do
    p=d3m.primitives.$suffix
    mkdir -p $isi/$p/$version
    echo "python -m d3m.index describe -i 4 $p > $isi/$p/$version/primitive.json"
    python -m d3m.index describe -i 4 $p > $isi/$p/$version/primitive.json
done
