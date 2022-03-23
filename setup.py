import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version = "0.28.5"
setuptools.setup(
    name="sherlockpipe", # Replace with your own username
    version=version,
    author="M. Dévora-Pajares & F.J. Pozuelos",
    author_email="mdevorapajares@protonmail.com",
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
    python_requires='>=3.6.8',
    install_requires=['astroplan==0.7',
                        "astroquery==0.4.1",
                        "allesfitter==1.2.8",
                        'argparse==1.4.0',
                        'beautifulsoup4==4.9.3', # Parsing HTML and XML, for OIs extraction
                        "celerite==0.4.0", # Allesfitter dependency
                        "corner==2.1.0", # Allesfitter dependency
                        "cython==0.29.21",
                        "dearwatson==0.1.12", # Vetting
                        "dynesty==1.0.1", # Allesfitter dependency
                        "ellc==1.8.5", # Allesfitter dependency
                        "emcee==3.0.2", # Allesfitter dependency
                        "mock==4.0.3",
                        'numba==0.53.1', # foldedleastsquares dependency
                        'pytz', # Observation plan: Not using version because it gets the DB updated with each release
                        "requests==2.25.1",
                        "rebound==3.17.3",
                        "reproject==0.4",
                        "seaborn==0.11.1",
                        'setuptools>=41.0.0',
                        "sklearn==0.0",
                        "spock==1.3.1",
                        'statsmodels==0.12.2', # Allesfitter dependency
                        'timezonefinder==5.2.0', # Observation plan
                        'tqdm==4.56.0',
                        'triceratops==1.0.10',
                        'uncertainties==3.1.5'
    ]
)