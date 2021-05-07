from setuptools import setup

setup(name="gym_lunarlanderhardcore",
      version="0.01",
      author="Niklas Kiehne",
      packages=["gym_lunarlanderhardcore", "gym_lunarlanderhardcore.envs"],

      install_requires = ["gym", "numpy"]
)