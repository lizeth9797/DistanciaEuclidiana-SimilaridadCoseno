import csv

with open('Tweets.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
    print(data)




