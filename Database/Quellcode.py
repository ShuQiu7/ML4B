#Das wird das Code Blatt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

#Daten einlesen
df1 = pd.read_csv("C:/Users/Felix/OneDrive/10_FAU/Semester 6/Machine Learning for Business/Datenblatt1.csv", encoding="ISO-8859-1")

# Alles außer a-z und A-Z entfernen
data1 = df1.iloc[:, 2:27]
data1.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# Spaltennamen in Zahlen ändern
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data1.columns = new_index

# Großbuchstaben in Kleinbuchstaben
for i in new_index:
    data1[i] = data1[i].str.lower()

# Daten in jeder Zeile in String konvertieren und zu einem String zusammenfassen
# => Nachrichten pro Tag in einem String zusammengefasst
headlines1 = []
for row in range(0, len(data1.index)):
    headlines1.append(' '.join(str(x) for x in data1.iloc[row, 0:25]))

# Aufteilen in Features (Nachrichten) und Labels
X = [item for item in headlines1]
Y = [item for item in df1.iloc[:, 1]]


