# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'SHERLOCK PIPEline'
copyright = '2021, Martín Dévora-Pajares & Francisco J. Pozuelos'
author = ' Martín Dévora-Pajares & Francisco J. Pozuelos'

sys.path.insert(0, os.path.abspath('../../'))

# The full version, including alpha/beta/rc tags
release = '0.26.0'
extensions = [
    "sphinxcontrib.mermaid",
    "sphinx_rtd_theme",
    "myst_nb",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_mock_imports = [
    'sherlockpipe.tests',
    'sherlockpipe.regression_tests',
]
#autosummary_imported_members = True

nb_execution_mode = "off"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None
