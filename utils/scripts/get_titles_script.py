"""
Purpose:
-----------------------------------------------------------------------------------
- What are the titles
-----------------------------------------------------------------------------------
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname('__file__'), "../functions/"))
from get_title import *

titles = titanic["Name"].apply(get_title)
print(titles.unique())

