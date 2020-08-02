---
title: 'The SHERLOCK PIPEline: Searching for Hints of Exoplanets fRom Light curves Of spaCe-based seeKers.'
tags:
  - Python
  - Astronomy
  - Exoplanets
  - Kepler
  - TESS
authors:
  - name: Francisco J. Pozuelos^[Custom footnotes for e.g. denoting who the corresspoinding author is can be included like this.]
    orcid: 0000-0003-1572-7707
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Martin Devora
    affiliation: 3
affiliations:
 - name: Space Sciences, Technologies and Astrophysics Research (STAR) Institute, Universit\'e de Li\`ege, All\'ee du 6 Ao\^ut 19C, B-4000 Li\`ege, Belgium
   index: 1
 - name: Astrobiology Research Unit, Universit\'e de Li\`ege, All\'ee du 6 Ao\^ut 19C, B-4000 Li\`ege, Belgium
   index: 2
 - name: Valencia International University
   index: 3
date: 2 August 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Transit detection of exoplanets is the one of the most fruitful methods to find planets beyond the Solar System. 
At the time of writting, more that 4,000 exoplanets have been discovered, most of them
by the transit method, that is, a planet passes in front the disk of its host star, 
blocking a fraction of the star light, and provoking a decrease of the observed flux. After the transit, the total
flux rises again to its nominal value.

The transit method was revolutionised by the Kepler (and its extended K2) mission, launched in 2009 and operating untill 2018, it discovered 
more than 2,600 confirmed planets, whose legacy keeps offering exciting results nowadays [@prawjual:2020; @quentin:2020]. Since 2018, another space-based satellite relieved Kepler in the hunt of transiting exoplanets; the TESS (Transiting Exoplanets Satellite Survey) mission [@ricker:2015]. As opposed to Kepler, TESS is a nearly all-sky survey that focuses
on the nearest and brightest stars, searching for planets well suited, notably, for future atmospheric characterization. After completes its
two-year prime mission in July 2020, TESS started its extended mission, which will lasts until 2022. During its first
two years of operation, TESS has released more than 2,000 TOIs (TESS Objects of Interest) and confirmed more than 50 planets. 

Taken toguether, both missions Kepler and TESS, offer a huge data base of high-quality continuos observations, with an excellent 
photometric precision, which allows the detection of exoplanets radii down to ~1R$_{\oplus}$.


# The SHERLOCK pipeline 

The SHERLOCK (Searching for Hints of Exoplanets fRom Light curves Of spaCe-based seeKers) PIPEline is a ready-to-use and user-friendly pipeline, which
minimizes the interaction of the user to the minimum. SHERLOCK is based on previous well-known and well-tested codes which allow the exoplanets community 
to explore the public data from Kepler and TESS missions without need of a deep knowledge of how the data are built and stored. In most of 
cases the user only needs to provide with a KOI-ID, EPIC-ID, TIC-ID or coordinates of the host star where wants to search for exoplanets. 
SHERLOCK makes use of lightkurve [@lightkurve:2018], wotan [@wotan:2019], eleonor [@eleonor:2019], and transit least squares [@tls:2019] packages to download, process, and search for exoplanets 
in any of the thousands of public light curves provided by Kepler and TESS missions. As output of the usage of SHERLOCK, it is printed a collection of
plots and log files which allow the user to explore the most promising signals. 


The SHERLOCK workflow is as follows: 

1. Data download.
The light curve where the user wants to search for exoplanets it is downloaded from the NASA Mikulski Archive for Space Telescope (MAST). In the case of the TESS data, it is used the 
Pre-search Data Conditioning Simple APerture (PDC-SAP) flux given by the SPOC (Science Process-ing  Operations  Center). For Kepler data, the Pre-search Data
Conditioning (PDC) given by SOC (Science Operations Center). In both cases, these light curves are corrected of systematic error sources such as pointing drift,
focus changes, and thermal transient. In the case of exploring the FFIs (Full-Frame Images) from TESS, it is taken by default the PCA (Principal Component Analysis) 
flux provided by eleonor, but the user can choose among the different data product available; raw, corrected, PCA, and PSF-modeled flux (see @eleonor:2019 for 
more details). 

2. Bad data selfmasking.
It happens sometimes that a particular region of a given light curve is very noisy, mainly due to a jitter in the spacecraft. This noisy region
might has a strong-negative impact in the performance of the pipeline, provoking that all the potential planets remain undetected. To overcome this issue 
SHERLOCK automaticaly sets a mask in these noise regions. To chose where has to be implemented the mask, it is computed the RMS in blocks of 4 hours, 
when the RMS is 2.0x over the mean value (default value), the data is masked. When the selfmasking funtion is acting, it is plotted a RMS Vs. Time graph, where the user
may decide which is the appropiate threshold to set correctly the mask. From our experience, threshold ranges from (1.25-2.0)x over the mean value
(see \autoref{fig:rms})
In addition, if the user does not agree with the mask proposed automatically by the Selfmasking function, it can be deactivated and set manually the mask, the user
only needs to provide the time to be masked. 

![Selfmasking function evaluates the RMS of the light curve in blocks of four hours and mask these regions where the RMS is 2.0x over the mean value .\label{fig:rms}](rms.png)


3. The search.

TODO

4. Exploring results. 

TODO

5. What's next?. 

Vetting and fit. 

--------------------- examples of the template from here to down --------------

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
```	

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
