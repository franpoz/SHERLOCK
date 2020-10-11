import setuptools
import os

with open("README.md", "r") as fh:
    long_description = fh.read()
version = os.popen('git tag -l --sort -version:refname | head -n 1').read().split('\n', 1)[0]

setuptools.setup(
    name="sherlockpipe", # Replace with your own username
    version=version,
    author="F.J. Pozuelos & M. Dévora",
    author_email="fjpozuelos@uliege.be",
    description="Search for Hints of Exoplanets fRom Lightcurves Of spaCe based seeKers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/franpoz/SHERLOCK",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["numpy",
                        "cython",
                        "pandas",
                        "lightkurve",
                        "transitleastsquares",
                        "requests",
                        "eleanor",
                        "wotan",
                        "matplotlib",
                        "pyyaml",
                        "allesfitter",
                        "seaborn",
                        "astroplan",
                        "astroquery",
                        "sklearn",
                        "scipy",
                        "tess-point",
                        "reproject==0.4",
                        "reportlab",
                        "astropy==3.1.1",
                        "mock > 2.0.0"
    ]
)
