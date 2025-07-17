import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("salary_model.pkl", "rb"))

# Tabs in Streamlit
tab1, tab2 = st.tabs([" Predict Salary", "Visualizations"])

# ---------- TAB 1: Salary Prediction ----------
with tab1:
    st.title("Employee Salary Prediction App")

   
    input_dict = {"age": 30,"workclass": 4,"fnlwgt": 200000,"education": 10,"educational-num": 10,"marital-status": 2,"occupation": 5,"relationship": 1,"race": 2,"gender": 1,"capital-gain": 0,"capital-loss": 0,"hours-per-week": 40,"native-country": 15,"age_group": 1  }

    # Update inputs
    input_dict["age"] = st.slider("Age", 17, 75, 30)
    input_dict["educational-num"] = st.slider("Education Number", 1, 16, 10)
    input_dict["hours-per-week"] = st.slider("Hours per Week", 1, 100, 40)
    input_dict["capital-gain"] = st.number_input("Capital Gain", 0, 100000, 0)
    input_dict["capital-loss"] = st.number_input("Capital Loss", 0, 100000, 0)

    # Age group logic
    age = input_dict["age"]
    if age < 30:
        input_dict["age_group"] = 0  
    elif age <= 55:
        input_dict["age_group"] = 1  
    else:
        input_dict["age_group"] = 2 

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict Salary"):
        prediction = model.predict(input_df)[0]
        label = ">50K" if prediction == 1 else "<=50K"
        st.success(f" Predicted Salary Category: {label}")

# ---------- TAB 2: Visualization ----------
with tab2:
    st.title("Salary Distribution Insights")

    df = pd.read_csv("adult 3.csv")

    def get_age_group(age):
        if age < 30:
            return "Young"
        elif age <= 55:
            return "Middle-aged"
        else:
            return "Senior"

    df['age_group'] = df['age'].apply(get_age_group)

    fig, ax = plt.subplots()
    df['income'].groupby(df['age_group']).value_counts().unstack().plot(kind='bar', ax=ax)
    plt.title("Income by Age Group")
    plt.ylabel("Number of People")
    st.pyplot(fig)
