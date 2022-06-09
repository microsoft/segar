from setuptools import setup, find_packages

setup(
    author='JAXRL',
    name='jaxrl',
    install_requires=[
        "tensorboardX",
        "flax",
        "jax",
        "tensorflow_probability",
        "tensorflow"],
    packages=find_packages(),
    version='0.1',
    include_package_data=True,
    package_data={'': ['*.np']})
