import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("student_data.csv")

X = df[["StudyHours", "Attendance", "PreviousMarks"]]
y = df["Result"]

model = DecisionTreeClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully")
