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
