#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='digit_rec',
      version='0.0.1',
      description='Digit recognition with convolutional neural networks.',
      author='akram zaytar',
      author_email='medakramzaytar@gmail.com',
      url='https://github.com/akramz/digit_rec',  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
      install_requires=[
            'pytorch-lightning'
      ],
      packages=find_packages()
      )

