import streamlit as sl

import pandas as pd
import numpy as np

import joblib
import altair as alt

#loading the previously saved model
pipe_lr=joblib.load(open(r"models\emotion_classifier_pipe_lr_21_july_2022.pkl","rb"))

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def predict_emotion(file):
    preds=pipe_lr.predict([file])

    return preds[0]

def get_pred_prob(file):
    probs=pipe_lr.predict_proba([file])

    return probs

def main():
    sl.title("Emotion Classifier App")
    menu=["Home"]
    choice=sl.sidebar.selectbox("Menu",menu)

    if choice=="Home":
        sl.subheader("Home\nEmotion Detection in Text")
        with sl.form(key='emotion-clf'):
            text=sl.text_area("Enter your text here:-")
            submit_text=sl.form_submit_button(label='Submit')

        if submit_text:
            col1,col2=sl.columns(2)

            predictions=predict_emotion(text)
            probabilities=get_pred_prob(text)

            with col1:
                sl.success("Original Text:-")
                sl.write(text)

                sl.success("Prediction:-")
                emoji_icon=emotions_emoji_dict[predictions]
                sl.write("{} : {}".format(predictions,emoji_icon))
                sl.write("with a probability of :- {}%".format(np.max(np.round(probabilities*100,2))))

            with col2:
                sl.success("Prediction Probability:-")
                #sl.write(probabilities)
                prob_df=pd.DataFrame(np.round(probabilities*100,2),columns=pipe_lr.classes_)
                prob_df=prob_df.T.reset_index()
                prob_df.columns=["Emotions","Probabilities"]
                sl.write(prob_df)

                fig = alt.Chart(prob_df).mark_bar().encode(x='Emotions',y='Probabilities',color='Emotions')
                sl.altair_chart(fig,use_container_width=True)

if __name__=='__main__':
    main()
