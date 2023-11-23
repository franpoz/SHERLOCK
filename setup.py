import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version = "0.40.1"
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
    python_requires='>=3.10',
    install_requires=['arviz==0.12.1', # Validation required (pytransit, from triceratops)
                        'astroplan==0.8',
                        "astroquery==0.4.6",
                        "alexfitter==1.2.12", # Fit
                        'argparse==1.4.0', # All modules
                        'beautifulsoup4==4.9.3', # Parsing HTML and XML, for OIs extraction
                        "celerite==0.4.0", # Allesfitter dependency
                        "corner==2.2.2", # Allesfitter dependency
                        "Cython==0.29.21",
                        "dearwatson==0.7.15", # Vetting
                        "dynesty==1.0.1", # Allesfitter dependency
                        "ellc==1.8.5", # Allesfitter dependency
                        "emcee==3.0.2", # Allesfitter dependency
                        "h5py==3.7.0", # Allesfitter dependency
                        "mock==4.0.3",
                        'pdf2image==1.16.2',
                        'pytransit==2.5.21', #Validation
                        'pytz', # Observation plan: Not using version because it gets the DB updated with each release
                        "requests==2.25.1", # OIs management
                        "rebound==3.17.3", # Stability
                        "reproject==0.4",
                        "seaborn==0.11.1",
                        'setuptools>=41.0.0',
                        "sklearn==0.0",
                        'statsmodels==0.13.5', # Allesfitter dependency, might conflict with lcbuilder dependency for autocorrelation
                        'timezonefinder==5.2.0', # Observation plan
                        'tqdm==4.56.0',
                        'triceratops==1.0.17', # Validation
                        'uncertainties==3.1.5' # Observation plan
    ]
)
