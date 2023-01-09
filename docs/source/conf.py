import pathlib
import sys
import os
for x in os.walk('../../src'):
  sys.path.insert(0, x[0])


sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
import sphinx_rtd_theme



# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Conference of Parties Model'
copyright = '2022, Anthony Couthures'
author = 'Anthony Couthures'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    "nbsphinx",
    'sphinxcontrib.tikz',
    'sphinxcontrib.bibtex',
]

master_doc = 'index'

exclude_patterns = []

# def skip(app, what, name, obj, would_skip, options):
#     if name == "__init__":
#         return False
#     return would_skip

# def setup(app):
#     app.connect("autodoc-skip-member", skip)

numpydoc_show_class_members = True
autodoc_preserve_defaults = True
numpydoc_class_members_toctree = False
autodoc_default_options = {
    'member-order': 'bysource',
}


autodoc_default_flags = ['members']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

def setup(app):
    app.add_css_file('css/my_theme.css')
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# BibRef

bibtex_bibfiles = ['CO2.bib']

bibtex_default_style = 'unsrt'

# TikZ

tikz_proc_suite = 'pdf2svg'

tikz_tikzlibraries = 'backgrounds, arrows, shapes,shapes.geometric,shapes.misc'


tikz_latex_preamble = r"""\tikzstyle{tikzfig}=[baseline=-0.25em,scale=0.5]

% these are dummy properties used by TikZiT, but ignored by LaTex
\pgfkeys{/tikz/tikzit fill/.initial=0}
\pgfkeys{/tikz/tikzit draw/.initial=0}
\pgfkeys{/tikz/tikzit shape/.initial=0}
\pgfkeys{/tikz/tikzit category/.initial=0}

% standard layers used in .tikz files
\pgfdeclarelayer{edgelayer}
\pgfdeclarelayer{nodelayer}
\pgfsetlayers{background,edgelayer,nodelayer,main}

% style for blank nodes
\tikzstyle{none}=[inner sep=0mm]

% include a .tikz file
\newcommand{\tikzfig}[1]{%
{\tikzstyle{every picture}=[tikzfig]
\IfFileExists{#1.tikz}
  {\input{#1.tikz}}
  {%
    \IfFileExists{./figures/#1.tikz}
      {\input{./figures/#1.tikz}}
      {\tikz[baseline=-0.5em]{\node[draw=red,font=\color{red},fill=red!10!white] {\textit{#1}};}}%
  }}%
}

% the same as \tikzfig, but in a {center} environment
\newcommand{\ctikzfig}[1]{%
\begin{center}\rm
  \tikzfig{#1}
\end{center}}

% fix strange self-loops, which are PGF/TikZ default
\tikzstyle{every loop}=[]

% Node styles
\tikzstyle{new style 0}=[fill={rgb,255: red,216; green,216; blue,216}, draw=black, shape=rectangle, minimum height=1.5em, minimum width=3em]
\tikzstyle{new style 1}=[fill=white, draw=black, shape=circle, tikzit shape=circle, align=center]

% Edge styles
\tikzstyle{new edge style 3}=[-, fill={rgb,255: red,254; green,255; blue,180}, dashed]
\tikzstyle{new edge style 0}=[->]
\tikzstyle{new edge style 1}=[-, fill={rgb,255: red,223; green,221; blue,255}, draw=black]
\tikzstyle{new edge style 2}=[-, fill={rgb,255: red,231; green,255; blue,202}, draw=black]
"""