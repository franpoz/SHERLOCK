import setuptools
from setuptools.command.install import install
import platform
import os
import shutil
version = "1.2.2"
import subprocess

with open("README.md", "r") as fh:
    long_description = fh.read()

class CustomInstall(install):
    def run(self):
        here = os.path.abspath(os.path.dirname(__file__))
        ellc_dir = os.path.join(here, 'sherlockpipe', 'ellc')
        print(f"=== Running make in {ellc_dir} ===")
        subprocess.check_call(['make', '-B'], cwd=ellc_dir)
        shutil.copy(ellc_dir + '/libellc.so', os.path.join(ellc_dir, 'ellc') + '/libellc.so')
        super().run()

setuptools.setup(
    name="sherlockpipe", # Replace with your own username
    version=version,
    author="M. Dévora-Pajares & F.J. Pozuelos",
    author_email="mdevorapajares@protonmail.com",
    description="Search for Hints of Exoplanets fRom Lightcurves Of spaCe based seeKers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/franpoz/SHERLOCK",
    packages=setuptools.find_packages() + ['sherlockpipe.ellc'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={'install': CustomInstall},
    python_requires='>=3.11',
    install_requires=[
                        'astroplan==0.10.1',
                        "alexfitter==1.2.19", # Fit
                        'argparse==1.4.0', # All modules
                        "celerite==0.4.3", # Allesfitter dependency
                        "corner==2.2.3", # Allesfitter dependency
                        "dearwatson==0.16.4", # Vetting
                        "dynesty==2.1.5", # Allesfitter dependency
                        "emcee==3.1.6", # Allesfitter dependency
                        "mock==4.0.3",
                        'pytz', # Observation plan: Not using version because it gets the DB updated with each release
                        "requests==2.32.3", # OIs management
                        "rebound==4.4.1", # Stability
                        "reproject==0.13.0",
                        "seaborn==0.13.2",
                        'setuptools>=41.0.0',
                        "sklearn==0.0",
                        'statsmodels==0.13.5', # Allesfitter dependency, might conflict with lcbuilder dependency for autocorrelation
                        'timezonefinder==5.2.0', # Observation plan
                        'tqdm==4.56.0'
    ]
)
