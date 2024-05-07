#Das wird das Code Blatt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#Daten einlesen
df1 = pd.read_csv("Datenblatt1")
df2 = pd.read_csv("Datenblatt2")

#Training und Test Daten separieren
train1 = df1[df1['Date'] < '20150101']
test1 = df1[df1['Date'] > '20141231']

train2 = df2[df2['Date'] < '20120801']
test2 = df2[df2['Date'] > '20120802']

# Alles außer a-z und A-Z entfernen
data1 = train1.iloc[:, 2:27]
data1.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

data2 = train2.iloc[:, 2:27]
data2.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# Spaltennamen in Zahlen ändern
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data1.columns, data2.columns = new_index

# Großbuchstaben in Kleinbuchstaben
for i in new_index:
    data1[i] = data1[i].str.lower()
    data2[i] = data2[i].str.lower()

# Daten in Zeile 1 in String konvertieren und zu einem String zusammenfassen
headlines1 = []
for row in range(0, len(data1.index)):
    headlines1.append(' '.join(str(x) for x in data1.iloc[row, 0:25]))

headlines2 = []
for row in range(0, len(data2.index)):
    headlines2.append(' '.join(str(x) for x in data2.iloc[row, 0:25]))

# Implementierung von BAG of WORDS
countvector = CountVectorizer(ngram_range = (2, 2))
traindataset1 = countvector.fit_transform(headlines1)
traindataset2 = countvector.fit_transform(headlines2)
