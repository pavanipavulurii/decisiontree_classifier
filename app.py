import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("🌳 Decision Tree Classification App")
st.write("Iris Dataset Classification using Decision Tree")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = iris.target

# ---------------------------------------------------
# SIDEBAR - HYPERPARAMETERS
# ---------------------------------------------------

st.sidebar.header("Hyperparameters")

criterion = st.sidebar.selectbox(
    "Select Criterion",
    ["gini", "entropy"]
)

max_depth = st.sidebar.slider(
    "Max Depth",
    min_value=1,
    max_value=10,
    value=3
)

splitter = st.sidebar.selectbox(
    "Select Splitter",
    ["best", "random"]
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.5,
    value=0.2
)

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    splitter=splitter,
    random_state=42
)

# TRAIN MODEL
model.fit(X_train, y_train)

# PREDICTION
y_pred = model.predict(X_test)

# ---------------------------------------------------
# ACCURACY
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Accuracy")
st.write(f"Accuracy Score: {accuracy:.2f}")

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    columns=iris.target_names,
    index=iris.target_names
)

st.dataframe(cm_df)

# ---------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------

st.subheader("📄 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
)

st.text(report)

# ---------------------------------------------------
# TREE VISUALIZATION
# ---------------------------------------------------

st.subheader("🌳 Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(15, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    ax=ax
)

st.pyplot(fig)

# ---------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------

st.subheader("📊 Dataset Preview")

df = X.copy()
df["target"] = y

st.dataframe(df.head())