import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt

dataframe_all = pd.read_csv(
    "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
num_rows = dataframe_all.shape[0]

counter_nan = datafram_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]

datafram_all = dataframe_all[counter_without_nan.keys()]


