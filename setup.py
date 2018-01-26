from setuptools import setup

setup(name='dsbox-featurizer',
      version='0.1.0',
      url='https://github.com/usc-isi-i2/dsbox-featurizer',
      maintainer_email='fanghaol@usc.edu',
      maintainer='Fanghao Luo',
      description='DSBox Featurization primitives',
      license='MIT',
      packages=['dsbox', 'dsbox.datapreprocessing', 'dsbox.datapreprocessing.featurizer'],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
          'scipy>=0.19.0', 'numpy>=1.11.1', 'pandas>=0.20.1',
          'python-dateutil>=2.5.2', 'six>=1.10.0', 'stopit',
          'scikit-learn>=0.18.0'
      ],
      keywords='d3m_primitive',
      entry_points = {
          'd3m.primitives': [
#              'dsbox.MultiTableFeaturization = dsbox.datapreprocessing.featurizer.multiTable:MultiTableFeaturization'
              'dsbox.RandomProjectionTimeSeriesFeaturization = dsbox.datapreprocessing.featurizer.timeseries:RandomProjectionTimeSeriesFeaturization'
          ],
      }
)
