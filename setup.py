import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


setuptools.setup(
    name="blackswan", # Replace with your own username
    version="0.1",
    author="Adithya MN",
    author_email="adithya.niranjan2412982@gmail.com",
    description="Realtime Streaming Anomaly-Detection on Massive Log Files",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/esowc/BlackSwan",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='<3.8',
    include_package_data=True,

)
