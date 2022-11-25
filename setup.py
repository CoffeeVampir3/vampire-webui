from setuptools import setup, find_packages

setup(
    name='vampire-webui',
    version='1.0.0',
    description='Vampire Webui.',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'diffusers'
    ],
)
