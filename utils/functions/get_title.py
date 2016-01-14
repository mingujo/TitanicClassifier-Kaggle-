"""
Purpose:
-----------------------------------------------------------------------------------
- Use Regular Expression to sort out the titles in 'Name'
-----------------------------------------------------------------------------------
"""
import re

def get_title(name):
    # Use a regular expression to search for a title.  
    # Capital + lower + dot
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

