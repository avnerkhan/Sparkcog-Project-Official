import csv
import pandas as pd

writer_file = open("Crime_Reports_Shortened.csv", "w", newline="")

with open("Crime_Reports.csv") as crime_large_csv:
    csv_writer = csv.writer(writer_file)
    csv_reader = csv.reader(crime_large_csv, delimiter=",")
    line_count = 0
    for row in csv_reader:
        csv_writer.writerow(row)
        line_count += 1
        if line_count == 1000:
            break

dataset = pd.read_csv("Crime_Reports_Shortened.csv")
print(len(dataset))
print(dataset.head())