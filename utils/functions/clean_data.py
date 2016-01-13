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

def clean_data(df):
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
    ## Start with 'Sex'
    sex = np.sort(df['Sex'].unique())
    
    sex_int = dict(zip(sex, range(0, len(sex) + 1)))

    # Transform Sex from a string to a integer
    df['Sex_Val'] = df['Sex'].map(sex_int).astype(int)
    
    ## 'Embarked'
    embarked = np.sort(df['Embarked'].unique())

    # Generate a mapping of Embarked from a string to a number representation        
    embarked_int = dict(zip(embarked, range(0, len(embarked) + 1)))
    
    # Transform Embarked from a string to dummy variables
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked_Val')], axis=1)
    
    # Fill in missing values of 'Embarked'
    # Because the vast majority of passengers embarked in 'S': 3, 
    # we assign the missing values in Embarked to 'S':
    if len(df[df['Embarked'].isnull()] > 0):
        df.replace({'Embarked_Val' : 
                       {embarked_int[np.nan] : embarked_int['S']}
                   }, 
                   inplace=True)
    
    ## 'Fare'
    # Fill in missing values of Fare with the mean
    if len(df[df['Fare'].isnull()] > 0):
        avg_fare = df['Fare'].mean()
        df.replace({ None: avg_fare }, inplace=True)
    
    # To keep Age in tact, make a copy of it called AgeFill 
    # that we will use to fill in the missing ages:
    df['AgeFill'] = df['Age']

    # Determine the Age typical for each passenger class by Sex_Val.  
    # We'll use the median instead of the mean because 'Age' histogram seems to be right skewed.
    df['AgeFill'] = df['AgeFill'] \
                        .groupby([df['Sex_Val'], df['Pclass']]) \
                        .apply(lambda x: x.fillna(x.median()))
            
    # Define a new feature FamilySize that is the sum of Parch + SipSp
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    # Drop the features(columns) we won't use: (Age->AgeFill, SibSp,Parch->FamilySize)
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','SibSp','Parch','PassengerId'], axis=1)

    return df