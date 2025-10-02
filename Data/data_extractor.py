import pandas as pd

# UCI Pima Indians Diabetes Dataset (direct CSV link via GitHub mirror)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Define column names (from UCI description)
cols = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

# Load dataset
df = pd.read_csv(url, names=cols)

df.to_csv("diabetes_dataset.csv", index=False)

print(df.head())
print(df.describe())
