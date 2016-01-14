"""
Purpose:
-----------------------------------------------------------------------------------
- Change strings ('Sex','Embarked','') to a number representation
- Transform 'Embarked' from a string to a number 
- Combine familysize ('Parch'+'SibSp')
- Fill in the missing values ('Embarked','Fare')
- Drop unnecessary features ('Cabin','Ticket','Name')
-----------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import operator
from get_title import *
from get_family_id import *


def clean_data(df,drop_passenger_id):
    """ Return the cleaned data frame which is ready to transform into array
 
    Parameters
    ----------
    df : data.frame
        initial dafa frame
    
    Returns
    -------
    df : data.frame
        behavioral data added the ratio column (ratio : gain/loss)

    """
    ################ Start with 'Sex'
    sex = np.sort(df['Sex'].unique())
    sex_int = dict(zip(sex, range(0, len(sex) + 1)))

    # Transform Sex from a string to a integer
    df['Sex_Val'] = df['Sex'].map(sex_int).astype(int)
    
    ################ 'Embarked'
    embarked = np.sort(df['Embarked'].unique())

    # Generate a mapping of Embarked from a string to a number representation        
    embarked_int = dict(zip(embarked, range(0, len(embarked) + 1)))
    
    df['Embarked_Val'] = df['Embarked'] \
                               .map(embarked_int) \
                               .astype(int)

    # Transform Embarked from a string to dummy variables
    #df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val')], axis=1)

    # Fill in missing values of 'Embarked'
    # Because the vast majority of passengers embarked in 'S': 3, 
    # we assign the missing values in Embarked to 'S':
    if len(df[df['Embarked'].isnull()] > 0):
        df.replace({'Embarked_Val' : 
                       {embarked_int[np.nan] : embarked_int['S']
                       }
                   }, 
                   inplace=True)

    ################ 'FamilySize'
    # Define a new feature FamilySize that is the sum of Parch + SipSp
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    ################ 'Fare'
    # Fill in missing values of Fare with the mean
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({ None: avg_fare }, inplace=True)
    
    ################ 'Fare_per_person'
    df['Fare_per_person']=df['Fare']/(df['FamilySize']+1)    

    # To keep Age in tact, make a copy of it called AgeFill 
    # that we will use to fill in the missing ages:
    df['AgeFill'] = df['Age']

    # Determine the Age typical for each passenger class by Sex_Val.  
    # We'll use the median instead of the mean because 'Age' histogram seems to be right skewed.
    df['AgeFill'] = df['AgeFill'] \
                        .groupby([df['Sex_Val'], df['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))

    ################ 'Title'
    titles = df['Name'].apply(get_title)

    title_mapping = {"Mr": 1, "Miss": 2, "Ms": 2, "Mrs": 3, "Master": 4, 
                    "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Capt" : 7, "Mlle": 8, 
                    "Mme": 8, "Don": 9, "Sir": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Dona": 10}

    # convert the titles
    for k,v in title_mapping.items():
        titles[titles == k] = v

    df['Title'] = titles


    ################ 'FamilyID'
    family_id_mapping = {}

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
                curr_id = max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1
            family_id_mapping[family_id] = curr_id
        return family_id_mapping[family_id]
    
    family_ids = df.apply(get_family_id, axis=1)

    # There are so many family ids, so we'll compress all of the families under 3 members into one code.
    family_ids[df["FamilySize"] < 3] = -1

    df["FamilyId"] = family_ids

    # Drop the features(columns) we won't use: (Age->AgeFill, SibSp,Parch->FamilySize)
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','SibSp','Parch'], axis=1)

    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)
    
    return df