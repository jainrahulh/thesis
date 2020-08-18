# -*- coding: utf-8 -*-
"""
    Dumping the model to be used for future predictions.
"""

from joblib import load
text = ["higher R in the North West and South West is an important part of moving towards a more localised approach to lockdown"]
pipeline = load("text_classification.joblib")
pipeline.predict(text)