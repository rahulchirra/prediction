import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
st.title("Liver Disease Prediction App")
data_path = "/mnt/data/liver_disease_prediction.ipynb"
data = pd.read_csv(data_path)

# Display the dataset
st.subheader("Dataset")
st.write(data.head())

# Preprocessing: Encoding categorical columns if needed
if 'Gender' in data.columns:
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])

# Handling missing values if any
data = data.dropna()

# Display EDA
st.subheader("Data Visualization")
if st.checkbox("Show Correlation Heatmap"):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot()

# Splitting the data
X = data.drop("Liver_Disease", axis=1)  # Assuming 'Liver_Disease' is the target
y = data["Liver_Disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2%}")

# Prediction interface
st.subheader("Make a Prediction")
input_data = {}
for column in X.columns:
    input_data[column] = st.number_input(f"Enter {column}", value=float(data[column].mean()))

input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write("Prediction: Liver Disease" if prediction[0] == 1 else "No Liver Disease")

# Run the app by executing 'streamlit run app.py'
