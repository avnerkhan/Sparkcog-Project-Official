# Data Mining Project
Joshua Deng, Avner Khan, Abhishek Khare, Kiran Raja

# Our Dataset
For this project our team decided to analyze the Austin Police Departments Crime Reports Dataset. By creating a supervised learning model based on features like Highest Offense Code, Census Tract, and Location Type our model is trying to predict whether an arrest is likely to be made or not so we can reccommend how much police resource should be allocated to a given crime.

# Instructions to Run Code
  - Download Code
  - Download Crime_Reports.csv from here: https://data.austintexas.gov/Public-Safety/Crime-Reports/fdj4-gpfu (Click Export, and then CSV)
  - ```sh
    #run csv_cleaner.py, make sure all files including csvs are in the same directory
    $ python csv_cleaner.py
    ```
  - Open Crime_Reports_Project.ipynb in Jupyter Notebook
  - Run each cell in Crime_Reports_Project.ipynb in order

# Included Files
  - Crime_Reports.csv - Uncleaned Source Data from APD Repository
  - Crime_Reports_Project.ipynb - Jupyter Notebook where we performed data cleaning as well as used Darwin to construct model
  - Crime_Reports_Project.html - html version of the above notebook
  - csv_cleaner.py - at first used to downsample data for inital testing, also adjusts relevant labels as explained in paper
  - amb_sdk/ - Darwin SDK