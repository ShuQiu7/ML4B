from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

#Daten einlesen
df1 = pd.read_csv("C:/Users/Felix/OneDrive/10_FAU/Semester 6/Machine Learning for Business/Datenblatt1.csv", encoding="ISO-8859-1")
df2 = pd.read_csv("C:/Users/Felix/OneDrive/10_FAU/Semester 6/Machine Learning for Business/Datenblatt2.csv", encoding="ISO-8859-1")

# Alles außer a-z und A-Z entfernen
data1 = df1.iloc[:, 2:27]
data1.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

data2 = df2.iloc[:, 2:27]
data2.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# Spaltennamen in Zahlen ändern
list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data1.columns = new_index
data2.columns = new_index

# Großbuchstaben in Kleinbuchstaben
for i in new_index:
    data1[i] = data1[i].str.lower()
    data2[i] = data2[i].str.lower()

# Daten in jeder Zeile in String konvertieren und zu einem String zusammenfassen
# => Nachrichten pro Tag in einem String zusammengefasstheadlines = []
headlines = []
for row in range(0, len(data1.index)):
    headlines.append(' '.join(str(x) for x in data1.iloc[row, 0:25]))

for row in range(0, len(data2.index)):
    headlines.append(' '.join(str(x) for x in data2.iloc[row, 0:25]))

# Aufteilen in Features (Nachrichten) und Labels
x = [item for item in headlines]
y = [item for item in df1.iloc[:, 1]] + [item for item in df2.iloc[:, 1]]

# TF-IDF-Vektorisierung
vectorizer = TfidfVectorizer()
x_tfidf = vectorizer.fit_transform(x)

# Aufteilen des Datensatzes in Trainings- und Testsets
x_train, x_test, y_train, y_test = train_test_split(x_tfidf, y, test_size=0.2, random_state=42)

# Trainieren eines Klassifikators (hier SVM)
clf = SVC(kernel='linear')
clf.fit(x_train, y_train)

# Vorhersagen auf dem Testset
y_pred = clf.predict(x_test)

# Auswertung der Vorhersagegenauigkeit
accuracy = accuracy_score(y_test, y_pred)
print("Vorhersagegenauigkeit:", accuracy)

