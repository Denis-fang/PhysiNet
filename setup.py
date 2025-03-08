from setuptools import setup, find_packages

setup(
    name="physinet",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.0',
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'opencv-python>=4.5.0',
        'PyYAML>=5.4.0',
        'tqdm>=4.50.0',
        'einops>=0.3.0'
    ]
) 