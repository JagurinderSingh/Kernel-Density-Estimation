# Import required libraries
import os
import pandas as pd
from dotenv import load_dotenv

#Load the variables from the .env file into Python's environment
load_dotenv()

#Access the variable using os.getenv()
folder_path_final = os.getenv("output_file_path_env") # Folder path where your yearly CSV files are located

final_dataframe = pd.read_csv(folder_path_final)

print(final_dataframe)

