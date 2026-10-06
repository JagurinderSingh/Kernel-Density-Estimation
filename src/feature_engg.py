# Import required libraries
import os
import pandas as pd
from dotenv import load_dotenv
import numpy as np

def load_final_dataframe():

  #Load the variables from the .env file into Python's environment
  load_dotenv()

  #Access the variable using os.getenv()
  folder_path_final = os.getenv("output_file_path_env") # Folder path where your yearly CSV files are located

  final_dataframe = pd.read_csv(folder_path_final)

  return final_dataframe


final_dataframe = load_final_dataframe()

final_dataframe["Daily Log Returns"] = np.log(final_dataframe["Close"] / final_dataframe["Close"].shift(1)) # .shift(1) pushes down each row exactly by one row side by side of original row and then one by one keeps on dividing the original close by the pushed down close (visualise this on paper) and uses the formula of (log(b-a)) = log b - log a where b represents the final value and a represents the initial value so it is (original value/shifted value)

# Sum of daily log returns over a rolling 21-day trading window which is equivalent to the a full trading month
final_dataframe["Rolling 21D Log Returns"] = final_dataframe["Daily Log Returns"].rolling(window=21).sum() # We are capturing the current row one by one and then picking up its respective last 20 trading days, resulting into the total of 21 trading days. We are not summing up forward 20 days to prevent lookahead bias as our model will see the "future" which is wrong, we can only evaluate or see the past.

print(final_dataframe[["Close", "Daily Log Returns", "Rolling 21D Log Returns"]])
