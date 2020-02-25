import csv
with open('Tweets.csv', 'r') as file:
    reader = csv.reader(file)
    numero_tweets = 0
    for row in reader:
          print(f'\nNumero de Tweet: {numero_tweets+1}\n {row[0]}')
          numero_tweets += 1
    print(f'\n\nTotal de Tweets: {numero_tweets}')
