import pandas as pd 
import numpy as np 

import streamlit as st 
import pickle as pk
import PIL as p

model_log=pk.load(open('model_logistic.pkl','rb'))

st.write('The Titanic disaster which occurred in 1912 remains as one of the biggest tragedies that occurred in human endeavours. The objective of this paper is to apply different algorithms to check whether a passenger survived the Titanic disaster based on different attributes a passenger possess which is included in the dataset for testing. The results from the application of the different algorithms are compared and analysed.')
titain_img=p.Image.open('titan_pic.JPG')
st.image(titain_img,width=100,use_column_width=True)
def titanic_data(Pclass,Sex ,Age ,SibSp ,Parch ,Fare ,Q_cabin ,S_cabin):
    data_input = {'Pclass':Pclass,'Sex':Sex,'Age':Age,'SibSp':SibSp,'Parch':Parch,'Fare':Fare,'Q_cabin':Q_cabin,'S_cabin':S_cabin}
    titanic_frame=pd.DataFrame(data_input,index=[0])
    prdd=model_log.predict(titanic_frame)
    pred_prob=model_log.predict_proba(titanic_frame)
    
    return prdd, pred_prob





def main():
    
    st.title("** ARTIFICIAL INTELLIGENCE  **")
    html="""<div style= "background-color:red" ;padding : 15px"">
        <h2> <b> --CHECKING SURVIVAL CHANCES-- </b> </h2>
    </div>
    """
    st.markdown(html,unsafe_allow_html=True)
    Pclass=st.selectbox('Passenger Class' , ['FIRST_CLASS','SECOND_CLASS','THIRD_CLASS'])
    if Pclass=='FIRST_CLASS':
        Pclass=1
    elif Pclass=='SECOND_CLASS':
        Pclass=2
    else:
        Pclass=3
    Sex=st.selectbox('Sex of The Passenger ',['MALE','FEMALE'])
    if Sex=='MALE':
        Sex=1
    else:
        Sex=0
    Age=st.text_input('Áge of the Passenger   ')
    SibSp=st.selectbox('Number of  Siblings/Spouse The Passenger Had on The ship ',[0,1,2,3,4,5,6,7,8])
    Parch=st.selectbox('Number of Parents/Children The Passenger Had on the ship  ',[0,1,2,3,4,5,6])
    Fare=st.text_input('Fare of the Passenger ')
    Q_cabin=st.selectbox('Passenger in Q_cabin ,',['YES','NO'])
    if Q_cabin=='YES':
        Q_cabin=1
    else:
        Q_cabin=0
    S_cabin=st.selectbox('Passenger in S_cabin',['YES','No'])
    if S_cabin=='YES':
        S_cabin=1
    else:
        S_cabin=0
    

    if st.button('PREDICT PASSENGER SURVIVAL'):
        data_titanic=titanic_data(Pclass,Sex ,Age ,SibSp ,Parch ,Fare ,Q_cabin ,S_cabin)[0]
        mdl=titanic_data(Pclass,Sex ,Age ,SibSp ,Parch ,Fare ,Q_cabin ,S_cabin)[1]
        a=np.round((mdl[0][0]) * 100,1)
        b=np.round((mdl[0][1]) * 100,1)
        if data_titanic[0]==1:
            st.write('Life Boat Jacket Emergency')
            st.success(' CONGRATULATIONS THIS PASSENGER SURVIVED',)
            st.write('Survival Probablity Chances: NO is {}% , YES is {}%'.format(a,b))
            
        else:
            die_pic=p.Image.open('rip.JPG')
            st.image(die_pic,caption='DIED')
            st.error(' SORRY THIS PASSENGER DIED ')
            st.write('Survival Probablity Chances: NO is {}% , YES is {}%'.format(a,b))
            
       
            
       
if __name__=='__main__':
    main()


st.title("!!! DISCLAIMER ")
st.write('This is an Artificail Intelligence And Its not Totally Accurate')
st.write('_______')
st.write('Machine Learning Model Developed By Emmanauel Oladejo')


st.header('Link to the Github Code')
st.write('https://github.com/EMMANUEL1111/Titanic_Prediction/blob/main/TITANIC%20DATA.ipynb')
