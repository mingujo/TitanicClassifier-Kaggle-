"""
Purpose:
-----------------------------------------------------------------------------------
- Get the family ID for each passenger. 
- family ID is in a form of "LASTNAME + FAMILYSIZE"
-----------------------------------------------------------------------------------
"""

import clean_data
import operator

def get_family_id(row):
    # Find the last name for a given row
    last_name = row["Name"].split(",")[0]

    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # ex) Braund1

    # Find the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            curr_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            curr_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]