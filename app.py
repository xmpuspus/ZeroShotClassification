import streamlit as st
import pandas as pd
from ktrain import text

# Header
### Set Title
st.title("Zero-shot Risk Predictor and Industry Classifier")
st.markdown("""Uses BART for natural language infence to classify text based on [this reference](https://arxiv.org/abs/1910.13461).""")

# Text input
doc = st.text_area("Input text:", """We collect data on an excel database back in our clinic.""", height=250)

# @st.cache(allow_output_mutation=True)
def load_model():
    model = text.ZeroShotClassifier()
    return model

# @st.cache(allow_output_mutation=True)
def nli_predict(doc, model, list_labels, nli_template):

    prediction = model.predict(doc, 
                labels=list_labels, 
                include_labels=True, 
                multilabel=False, nli_template=nli_template)

    # Reformat to dataframe
    label_df = (pd.DataFrame(prediction)
              .rename(columns={0:'Label', 1:"Prediction Score"})
              .sort_values(by="Prediction Score", ascending=False))

    # Filter for significance (1/num_risks)
    label_filtered_df = label_df[label_df['Prediction Score']>= 1/(len(list_labels))]
    return label_filtered_df

# Model Parameters

# Input topics
risk_labels = ['Data Collection', 
               'Data Privacy', 
               'Data Processes', 
               'Data Modelling', 
               'Data Encryption',
               "Data Security", 
               "Not Related to Data"]

# industry_labels = ['Aerospace',
#                  'Transport',
#                  'Computer',
#                  'Telecommunication',
#                  'Agriculture',
#                  'Construction',
#                  'Education',
#                  'Pharmaceutical',
#                  'Food',
#                  'Healthcare',
#                  'Hospitality',
#                  'Entertainment',
#                  'News Media',
#                  'Energy',
#                  'Manufacturing',
#                  'Music',
#                  'Mining',
#                  'Internet',
#                  'Electronics',
#                  'Others']

industry_labels = [
                 'Telecommunication',
                 'Agriculture',
                 'Construction',
                 'Education',
                 'Pharmaceutical',
                 'Food',
                 'Healthcare',
                 'News Media',
                 'Others']

# Natural Language Inference Premise inputs
nli_template_risk_label = "The risk factor of this text is on {}."

nli_template_industry_label = "The industry described in the text is {}."

model = load_model()

st.subheader('Risk Predictions')
risk_predictions = nli_predict(doc, model, risk_labels, nli_template_risk_label)
st.write(risk_predictions)

st.subheader('Affected Industries')
risk_predictions = nli_predict(doc, model, industry_labels, nli_template_industry_label)
st.write(risk_predictions)