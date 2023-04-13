import torch
import pandas as pd
from MyLogisticRegression import MyLogisticRegression
import math


def load_data(file_name: str):
    df = pd.read_csv(file_name)
    df = df.drop('id', axis=1)
    df = df.dropna(how='any')
    
    # Clean Gender
    df.loc[df['gender'] == 'Male', 'gender'] = 0  # Male = 0
    df.loc[df['gender'] == 'Female', 'gender'] = 1  # Female = 1
    df.loc[df['gender'] == 'Other', 'gender'] = 2  # Female = 1
    
    # Clean Ever Married
    df.loc[df['ever_married'] == 'Yes', 'ever_married'] = 1
    df.loc[df['ever_married'] == 'No', 'ever_married'] = 0
    
    # Clean Residence Type
    df.loc[df['Residence_type'] == 'Urban', 'Residence_type'] = 1
    df.loc[df['Residence_type'] == 'Rural', 'Residence_type'] = 0
    
    # Clean smoking_status
    df.loc[df['smoking_status'] == 'never smoked', 'smoking_status'] = 0
    df.loc[df['smoking_status'] == 'formerly smoked', 'smoking_status'] = 1
    df.loc[df['smoking_status'] == 'smokes', 'smoking_status'] = 2
    df.loc[df['smoking_status'] == 'Unknown', 'smoking_status'] = 3
    
    # Clean work_type
    df.loc[df['work_type'] == 'children', 'work_type'] = 0
    df.loc[df['work_type'] == 'Govt_job', 'work_type'] = 1
    df.loc[df['work_type'] == 'Never_worked', 'work_type'] = 2
    df.loc[df['work_type'] == 'Private', 'work_type'] = 3
    df.loc[df['work_type'] == 'Self-employed', 'work_type'] = 4
    
    Y = df['stroke'].to_numpy()
    X = df.iloc[:, :-1].to_numpy()
    return X, Y


if __name__ == '__main__':
    X, Y = load_data('db/healthcare-dataset-stroke-data.csv')
    logistic_reg = MyLogisticRegression(X, Y)
    accuracy = logistic_reg.train()
    print(accuracy)