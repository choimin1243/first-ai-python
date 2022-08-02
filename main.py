import cv2
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time
import streamlit as st
import threading

st.experimental_memo.clear()
st.experimental_singleton.clear()


agree = st.checkbox('start?')

if agree:
    st.experimental_memo.clear()
    st.experimental_singleton.clear()
    st.write('Great!')




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
A=[-175, -199, -175, -107, -13, 84, 160, 198, 186, 130, 41, -57, -142, -192, -195]
B=[-95, 0, 96, 168, 199, 181, 118, 27, -70, -151, -195, -191, -140, -55, 43]
C=[59, -33, -31, 59, -18, -44, 55, -1, -53, 46, 15, -59, 33, 30, -59]
D=[54, 50, 0, -50, -54, -7, 45, 57, 15, -40, -59, -23, 33, 59, 31]

image_container={"img":0}

    


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
        st.experimental_memo.clear()
        st.experimental_singleton.clear()


        for i in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image,i,mp_hands.HAND_CONNECTIONS)
            x8,y8=int(i.landmark[8].x*640), int(i.landmark[8].y*480)
            second=time.strftime('%S')
            second=int(second)
            cv2.circle(image, (x8,y8),20, (0,0,255), -1)
            cv2.circle(image,(x8+A[image_container["img"]%15],y8+B[image_container["img"]%15]),8,(255,0,0),-1)
            cv2.circle(image,(x8+A[image_container["img"]%15]+C[image_container["img"]%15],y8+B[image_container["img"]%15]+D[image_container["img"]%15]),3,(0,255,255),-1)
            image_container["img"]=image_container["img"]+1
            time.sleep(0.2)
            print(image_container["img"])

                

    return cv2.flip(image, 1)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    st.experimental_memo.clear()
    st.experimental_singleton.clear()
    def recv(self, frame):
        st.experimental_memo.clear()
        st.experimental_singleton.clear()
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
