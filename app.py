import streamlit as st
import pickle
import numpy as np
# Load the trained model
model = pickle.load(open('model_admission.pkl', 'rb'))
st.title('Prediction admission')
st.info('Pour predire admission des etudiants en fonction de leurs heures d\'étude et de leurs scores de test')
# Get user input
heureEtude = st.slider('Heures d\'étude',0,11)
scoreTest = st.number_input('Score de test', min_value=0,max_value=100)
if st.button('Predire'):
    data = np.array([[heureEtude, scoreTest]])
    prediction = model.predict(data)
    proba= model.predict_proba(data)
    if prediction[0] == 1:
        st.success('l\'etudiant va reussir')
    else:        
        st.error('l\'etudiant va echouer')
    st.write('Probabilité de réussite:', proba[0][1])
    st.write('Probabilité d\'échec:', proba[0][0])