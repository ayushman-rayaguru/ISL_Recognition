#from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.models import load_model
from PIL import Image,ImageEnhance
import streamlit as st
import numpy as np
#import argparse
#import time
#import glob
#import random
import cv2
#
import mediapipe as mp 
import joblib

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(suppress_st_warning=True)
@st.cache(persist=True)



def data_clean(landmark):
  
  data = landmark[0]
  
  try:
    data = str(data)

    data = data.strip().split('\n')

    garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

    without_garbage = []

    for i in data:
        if i not in garbage:
            without_garbage.append(i)

    clean = []

    for i in without_garbage:
        i = i.strip()
        clean.append(i[2:])

    for i in range(0, len(clean)):
        clean[i] = float(clean[i])

    
    return([clean])

  except:
    return(np.zeros([1,63], dtype=int)[0])


# helper function to predict in real-time
def videopreds():
    hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)

    @st.cache(allow_output_mutation=True)
    def get_cap():
        return cv2.VideoCapture(0)

    cap = get_cap()

    frameST = st.empty()
    while cap.isOpened():
        success, image = cap.read()
  
        image = cv2.flip(image, 1)
  
        if not success:
            break

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cleaned_landmark = data_clean(results.multi_hand_landmarks)
            #print(cleaned_landmark)

            if cleaned_landmark:
                clf = joblib.load('model.pkl')
                y_pred = clf.predict(cleaned_landmark)
                image = cv2.putText(image, str(y_pred[0]), (50,150), cv2.FONT_HERSHEY_SIMPLEX,  3, (0,0,255), 2, cv2.LINE_AA) 
    
        #changes#cv2.imshow('MediaPipe Hands', image)
        frameST.image(image, channels="BGR")
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


@st.cache(persist=True)
def load_image(img):
    im = Image.open(img)
    return im


ig = Image.open('Heading.png')
st.image(ig, width=600)

st.title('Indian Sign Language Alphabet Recognition')



def main():

	menu = ['About us', 'Learning Module', 'Teaching Module']
	choice = st.sidebar.selectbox('Menu',menu)

	if choice == 'Image Detection':
		st.subheader('**ISL Detection**')
		img_file_buffer = st.file_uploader("Upload an image")
		
		if img_file_buffer is not None:
			image = Image.open(img_file_buffer)
			img_array = np.array(image)
			imagepreds(img_array)

	elif choice == 'Learning Module':
		st.subheader("**ISL detection**")
		st.text("Make sign gestures using manual")
		videopreds()

	elif choice == 'About us':
		st.subheader('**Principal : DR.R.Sreemathy**')
		st.subheader('**Project Guide: DR.MP Turuk**')
		st.text('Engineers: Ayushman Rayaguru, Sampreeti Saha, Kiransingh Pal')
		st.text('Build with Streamlit,Mediapipe, Keras and OpenCV')
		st.header("**Select the options from sidebar: **")
		st.subheader("Learning Module: For opening the webcam and checking the results.")
		st.subheader("Teaching Module: Learn how to make signs.")
		st.subheader("Happy Learning")
		st.subheader("Creator: GRP 55")
		st.markdown('----------------- MADE BY PICT ---------------------')

# driver code   code\DataScience\BEP3			st.write("Predicted Sign:"+ str(y_pred[0])	
if __name__ == '__main__':
    main()
