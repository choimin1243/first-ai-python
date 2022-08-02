import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time
import math
import streamlit as st

st.experimental_memo.clear()
st.experimental_singleton.clear()

@st.experimental_singleton
def my_drawing():
    mp_drawing=mp.solutions.drawing_utils
    return mp_drawing


@st.experimental_singleton
def my_hands():
    mp_hands=mp.solutions.hands
    return mp_hands


@st.experimental_singleton
def hands():
    hands= mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


    return hands


mp_drawing= my_drawing()
mp_hands=my_hands()




hands = hands()
    


@st.experimental_memo
def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    if results.multi_hand_landmarks:
        for i in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image,i,mp_hands.HAND_CONNECTIONS)
            x8,y8=int(i.landmark[8].x*640), int(i.landmark[8].y*480)
            second=time.strftime('%S')
            second=int(second)
            cv2.circle(image, (x8,y8),20, (0,0,255), -1)
            earth_x=x8+int(200*math.sin(second/2))
            earth_y=y8+int(200*math.cos(second/2))
            moonx=earth_x+20+int(60*math.sin(2*second+10))
            moony=earth_y+20+int(60*math.sin(2*second+10))
            cv2.circle(image,(earth_x,earth_y),8,(255,0,0),-1)
            cv2.circle(image,(moonx,moony),3,(0,255,255),-1)
                

                



    return cv2.flip(image, 1)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img= process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
