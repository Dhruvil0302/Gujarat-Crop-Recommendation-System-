import streamlit as st
import pickle
import numpy as np



DT_model=pickle.load(open('model_DT.pkl', 'rb'))
RF_model=pickle.load(open('model_RF.pkl', 'rb'))
KNN_model=pickle.load(open('model_KNN.pkl', 'rb'))
ET_model=pickle.load(open('model_ET.pkl', 'rb'))



def main():
    st.title("Created By Dhruvil M. Modi")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Crop Yield Recommendation System</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Decision Tree','Random Forest','K-Neighbors','Extra Tree']
    option=st.sidebar.selectbox('Which model you use?',activities)
    st.subheader(option)
    nt=st.slider('Select Nitrogen', 0, 200)
    ps=st.slider('Select Phosporus', 0, 200)
    po=st.slider('Select Potassium', 0, 200)
    tm=st.slider('Select Temperature', 0.0, 50.0)
    hu=st.slider('Select Humidity in %', 0.0, 100.0)
    ph=st.slider('Select Ph', 0.0, 10.0)
    rn=st.slider('Select Rainfall in mm', 0.0, 300.0)
    
    feature_list=[nt,ps,po,tm,hu,ph,rn]
    single_pred = np.array(feature_list).reshape(1,-1)
    if st.button('Recommend'):
        if option=='Decision Tree':
            st.success(DT_model.predict(single_pred))
        elif option=='Random Forest':
            st.success(RF_model.predict(single_pred))
        elif option=='K-Neighbors':
            st.success(KNN_model.predict(single_pred))
        else:
            st.success(ET_model.predict(single_pred))


if __name__=='__main__':
    main()