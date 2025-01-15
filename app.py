import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# set page config and title
st.set_page_config(page_title="💯Admission Score Predictor", layout="wide")
st.title("💯Admission Score Predictor")

# load the model and feature names
@st.cache_resource
def load_files():
    with open('multi_reg.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('feature_names.pkl', 'rb') as file:
        feature_names = pickle.load(file)
    return model, feature_names

# feature descriptions
feature_descriptions = {
    'univ_ranking': "University's global ranking position (lower number means better ranking)",
    'motiv_letter_strength': "Strength of motivation letter (scale: 1-5)",
    'recommendation_strength': "Strength of recommendation letters (scale: 1-5)",
    'gpa': "Grade Point Average (scale: 5-10)",
    'research_exp': "Research experience (0 = No, 1 = Yes)"
}

try:
    model, feature_names = load_files()
    feature_names = [f for f in feature_names if f != 'const'] # remove const

    # create two columns
    col1, col2 = st.columns([2,1])

    with col1:
        st.header('Enter Feature Values')
        # create input fields dynamically based on feature names
        input_dict = {}
        for feature in feature_names:
            input_dict[feature] = st.number_input(
                f"{feature}",
                value=0.0,
                help=feature_descriptions.get(feature, f"Enter value for {feature}")
            )

    # create a prediction button
    if st.button('Predict', type='primary'):
        # convert input to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Make prediction
        prediction = model.predict(input_df)
        
        with col2:
            st.header('Prediction Results')
            st.metric(label="Predicted Value", value=f"{prediction[0]:.2f}")
            
            # radar chart of inputs
            categories = list(input_dict.keys())
            values = list(input_dict.values())

            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values)]
                    )),
                showlegend=False,
                title="Input Features Distribution"
                )
            st.plotly_chart(fig, use_container_width=True)

        # feature importance bar chart
        st.header("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': abs(model.coef_)
        })
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(importance_df, 
                        x='Importance', 
                        y='Feature',
                        orientation='h',
                        title='Feature Importance (Absolute Coefficients)')
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
        
         # Optional: Display input values
        st.subheader("Input Summary")
        for feature, value in input_dict.items():
            st.text(f"{feature}: {value}")

    # Add information about the model
    with st.expander("About this Model"):
        st.write("""
        This is a multiple linear regression model that predicts Admission Scores based on the input features.
        The model was trained on historical data and can be used to make predictions for new inputs.
        """)
        
        # You can add more model details here
        st.write("Feature Names:", feature_names)

except FileNotFoundError:
    st.error("Please ensure model.pkl and feature_names.pkl are in the same directory as this script")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")

# Add custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin-top: 20px;
    }
    .streamlit-expanderHeader {
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)