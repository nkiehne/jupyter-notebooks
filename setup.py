from setuptools import setup
from setuptools import find_packages

long_description = '''Jupyter Notebook repository for several different topics
'''

setup(name='jupyter-notebooks',
      version='0.0.1',
      description='',
      long_description=long_description,
      author='Niklas Kiehne',
      author_email='niklas.kiehne@gmail.com',
      url='www.google.de',
      download_url='www.google.de',
      license='MIT',
      install_requires=["rise",
                        "notebook",
                        "numpy",
                        "pandas",
                        "matplotlib"],
      extras_require={},
      classifiers=[],
      packages=find_packages())