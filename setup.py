from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from .entry_points import entry_points

with open('requirements.txt', 'r') as f:
    install_requires = list()
    dependency_links = list()
    for line in f:
        re = line.strip()
        if re:
            install_requires.append(re)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # This is used so that it will pass d3m metadata submission process. The dsbox-featurizer package depends on
        import subprocess
        result = subprocess.check_output(['pip', 'list'])
        lines = str(result).split('\\n')
        for line in lines[2:]:
            part = line.split()
            if 'dsbox-featurizer' in part[0]:
                print(line)
                if '0' == part[1].split('.')[0]:
                    subprocess.call(['pip', 'uninstall', '-y', 'dsbox-featurizer'])
        import keras.applications.resnet50 as resnet50
        import keras.applications.vgg16 as vgg16
        resnet50.ResNet50(weights='imagenet')
        vgg16.VGG16(weights='imagenet', include_top=False)
        install.run(self)

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        import keras.applications.resnet50 as resnet50
        import keras.applications.vgg16 as vgg16
        resnet50.ResNet50(weights='imagenet')
        vgg16.VGG16(weights='imagenet', include_top=False)
        develop.run(self)

setup(name='dsbox-primitives',
      version='1.0.0',
      description='DSBox data processing primitives for both cleaning and featurizer',
      author='USC ISI',
      url='https://github.com/usc-isi-i2/dsbox-primitives.git',
      maintainer_email='kyao@isi.edu',
      maintainer='Ke-Thia Yao',
      license='MIT',
      packages=[
                'dsbox',
                'dsbox.datapreprocessing',
                'dsbox.datapreprocessing.cleaner', 
                'dsbox.datapostprocessing',
                'dsbox.datapreprocessing.featurizer',
                'dsbox.datapreprocessing.featurizer.multiTable',
                'dsbox.datapreprocessing.featurizer.image',
                'dsbox.datapreprocessing.featurizer.pass',
                'dsbox.datapreprocessing.featurizer.timeseries'
               ],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=install_requires,
      keywords='d3m_primitive',
      entry_points={
          'd3m.primitives': entry_points,
      },
      cmdclass={
          'develop': PostDevelopCommand,
          'install': PostInstallCommand
      })
