"""politeness-py: a Python port of the R package `politeness`
by Mike Yeomans, Alejandro Kantor and Dustin Tingley
(https://github.com/myeomans/politeness, MIT licence).

Main entry point: :func:`politeness`. See HOW_IT_WORKS.md for details.
"""
from . import datasets, glmnet
from .core import politeness
from .dnm import politenessDNM
from .examples import exampleTexts, findPoliteTexts
from .features import describe, feature_dictionary
from .models import (foldset, politenessModel, politenessProjection, receptive_model,
                     receptiveness, slogodds, trainModel)
from .plots import (featurePlot, featurePlotData, modelPlot, modelPlotData,
                    politenessPlot)
from .spacy_parser import spacy_initialize, spacyParser
from .text_tools import (cleanpunct, cleantext, ctxpand, dictWrap, noLeadDash,
                         textcounter, usWords)
from .token_sets import getTokenSets

__version__ = "0.1.0"
__r_package_version__ = "0.9.5"  # version of the R package this ports

__all__ = [
    "politeness", "politenessDNM", "feature_dictionary", "describe", "politenessModel", "receptiveness", "receptive_model",
    "trainModel", "politenessProjection", "featurePlot", "featurePlotData", "politenessPlot",
    "modelPlot", "modelPlotData", "exampleTexts", "findPoliteTexts", "slogodds", "foldset",
    "spacy_initialize", "spacyParser", "getTokenSets", "cleanpunct", "cleantext", "ctxpand",
    "noLeadDash", "dictWrap", "textcounter", "usWords", "datasets", "glmnet",
]
