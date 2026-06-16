import pandas as pd

def get_X_y(df) : 

    X = df.drop(["Cover_Type"] , axis = 1)
    y = df["Cover_Type"]

    y = y - 1
    
    return X , y

