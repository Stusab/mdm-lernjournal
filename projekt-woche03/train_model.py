import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Daten laden
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Modell trainieren
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Modell testen
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Modell-Accuracy: {accuracy:.2f}")

# Modell speichern
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modell wurde erfolgreich gespeichert unter model.pkl")
