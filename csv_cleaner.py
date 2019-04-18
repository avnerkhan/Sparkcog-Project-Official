import csv
import pandas as pd

writer_file = open("Crime_Reports_Fixed.csv", "w", newline="", encoding="utf-8")

with open("Crime_Reports.csv", encoding="utf-8") as crime_large_csv:
    csv_writer = csv.writer(writer_file)
    csv_reader = csv.reader(crime_large_csv, delimiter=",")
    line_count = 0
    for row in csv_reader:
        if (line_count > 0):
            if (row[18] != 'C' and row[18] != ''):
                # print('Adjusting status: ', row[18])
                row[18] = 'N'
        csv_writer.writerow(row)
        # print(row, line_count)
        line_count += 1
        # if line_count == 1000:
        #     break

dataset = pd.read_csv("Crime_Reports_Fixed.csv")
print(len(dataset))