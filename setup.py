from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="deconver",
    version="0.0.2",
    author="Pooya Ashtari",
    author_email="pooya.ash@gmail.com",
    description="Deconver - PyTorch",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pashtari/deconver",
    project_urls={
        "Bug Tracker": "https://github.com/pashtari/deconver/issues",
        "Source Code": "https://github.com/pashtari/deconver",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    keywords=[
        "machine learning",
        "deep learning",
        "image segmentation",
        "medical image segmentation",
        "deconvolution",
        "Deconver",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch", "torchvision", "einops"],
)
