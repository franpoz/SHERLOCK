import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sherlockpipe", # Replace with your own username
    version="0.9.2",
    author="Pozuelos",
    author_email="fjpozuelos@uliege.be",
    description="Search for Hints of Exoplanets fRom Lightcurves Of spaCe based seeKers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/franpoz/SHERLOCK",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    install_requires = ["numpy",
                        "cython",
                        "pandas",
                        "lightkurve",
                        "transitleastsquares",
                        "requests",
                        "eleanor",
                        "wotan",
                        "matplotlib"
    ]
)
