"""
Purpose:
-----------------------------------------------------------------------------------
- Random Forest Classifier Object
-----------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import pylab as plt
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)