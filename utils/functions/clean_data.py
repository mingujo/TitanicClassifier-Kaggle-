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
from sklearn import preprocessing


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

    #### for normalization
    le = preprocessing.LabelEncoder()

    ################ 'Sex'
    le.fit(df['Sex'] )
    x_sex=le.transform(df['Sex'])
    df['Sex']=x_sex.astype(np.float)

    ################ 'Embarked'
    embarked = np.sort(df['Embarked'].unique())

    # Generate a mapping of Embarked from a string to a number representation        
    embarked_int = dict(zip(embarked, range(0, len(embarked) + 1)))
    
    # df['Embarked_Val'] = df['Embarked'] \
    #                            .map(embarked_int) \
    #                            .astype(int)

    # Transform Embarked from a string to dummy variables
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val')], axis=1)

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
    

    ################ 'Title'
    titles = df['Name'].apply(get_title)

    title_mapping = {"Mr": 1, "Miss": 2, "Ms": 2, "Mrs": 3, "Master": 4, 
                    "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Capt" : 7, "Mlle": 8, 
                    "Mme": 8, "Don": 9, "Sir": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Dona": 10}

    # convert the titles
    for k,v in title_mapping.items():
        titles[titles == k] = v

    df['Title'] = titles

    ################ 'AgeFill'
    # To keep Age in tact, make a copy of it called AgeFill 
    # that we will use to fill in the missing ages:
    df['AgeFill'] = df['Age']

    # Determine the Age typical for each passenger class by Title.
    # We'll use the median instead of the mean because 'Age' histogram seems to be right skewed.
    age_by_title = df.groupby('Title')['AgeFill'].agg(np.median).to_dict()
    df['AgeFill'] = df.apply(lambda row: age_by_title.get(row['Title']) 
                     if pd.isnull(row['AgeFill']) else row['AgeFill'], axis=1)


    ############### 'AgeCat'
    df['AgeCat']=df['AgeFill']
    df.loc[ (df.AgeFill<=10) ,'AgeCat'] = 'child'
    df.loc[ (df.AgeFill>60),'AgeCat'] = 'aged'
    df.loc[ (df.AgeFill>10) & (df.AgeFill <=30) ,'AgeCat'] = 'adult'
    df.loc[ (df.AgeFill>30) & (df.AgeFill <=60) ,'AgeCat'] = 'senior'

    le.fit(df['AgeCat'])
    x_age=le.transform(df['AgeCat'])
    df['AgeCat'] =x_age.astype(np.float)


    ################ 'Fare'
    # Fill in missing values of Fare with the mean
    fare_by_pclass = df[df['Fare'] > 0].groupby('Pclass')['Fare'].agg(np.median).to_dict()
    df['Fare'] = df.apply(lambda r: r['Fare'] if r['Fare'] > 0 
                      else fare_by_pclass.get(r['Pclass']), axis=1)
    
    ################ 'Fare_per_person'
    df['Fare_per_person']=df['Fare']/(df['FamilySize']+1)    

    ################ 'HighLow'
    df['HighLow']=df['Pclass']
    df.loc[ (df.Fare_per_person<8.6) ,'HighLow'] = 'Low'
    df.loc[ (df.Fare_per_person>=8.6) ,'HighLow'] = 'High'

    le.fit(df['HighLow'])
    x_hl=le.transform(df['HighLow'])
    df['HighLow']=x_hl.astype(np.float)


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

    ################ 'Age_class'
    df["Age_class"] = df["Pclass"]*df["AgeFill"]


    ################ 'Fare_class'
    df['Fare_class']=df['Pclass']*df['Fare_per_person']


    ################ 'Ticket'
    le.fit( df['Ticket'])
    x_Ticket=le.transform( df['Ticket'])
    df['Ticket']=x_Ticket.astype(np.float)


    ################ 'Family'
    df['Family']=df['SibSp']*df['Parch']
    


    # ################ Sex_class
    # df["Sex_class"] = df["Pclass"]*df["Sex_Val"]

    # ################ AgeFill_squared
    # df["AgeFill_squared"] = df["AgeFill"]**2

    # ################ Age_class_squared
    # df["Age_class_squared"] = df["Age_class"]**2


    # Drop the features(columns) we won't use: (Age->AgeFill, SibSp,Parch->FamilySize)
    df = df.drop(['Name', 'Cabin', 'Embarked','Age','SibSp','Parch'], axis=1)

    if drop_passenger_id:
        df = df.drop(['PassengerId'], axis=1)
    
    return df