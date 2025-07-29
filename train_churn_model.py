import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv("Customer Churn.csv")  

print(data.head())

# Preprocessing
X = data.drop(columns=["Churn"])  
y = data["Churn"]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as 'churn_model.pkl'")
