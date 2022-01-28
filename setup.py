from setuptools import setup, find_packages

setup(
    name='rpp',
    install_requires=[
        "absl-py",
        "numpy",
        "scipy",
        "aenum",
        "matplotlib",
        "opencv-python-headless",
        "scikit-image",
        "sklearn",
        "h5py",
        "torch",
        "torchvision",
        "tqdm",
        "gym",
        "POT",
        "wandb"
    ],
    extras_require={
        'rl': [
            "ray[default]",
            "ray[rllib]",
        ]
    },
    packages=find_packages(),
    version='0.1a',
    include_package_data=True,
    package_data={'': ['*.np']})
