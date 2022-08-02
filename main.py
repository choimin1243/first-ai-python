import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
import time
import math
from streamlit import caching
import streamlit as st


@st.cache
def load_model1():
	mp_drawing = mp.solutions.drawing_utils
	return mp_drawing



@st.cache
def load_model2():
	mp_drawing_styles = mp.solutions.drawing_styles
	return mp_drawing_styles


@st.cache
def load_model3():
	mp_drawing_styles = mp.solutions.drawing_styles
	return mp_drawing_styles


@st.cache
def load_model4():
	mp_hands = mp.solutions.hands
	return mp_hands

@st.cache
def load_model5():
	hands = mp_hands.Hands(
    	model_complexity=0,
    	min_detection_confidence=0.5,
    	min_tracking_confidence=0.5)
	return hands



@st.cache
def load_model5():
	img_container={"time_init":True,"ml":150,"max_x":400,"max_y":50,"prev_x":0,"prev_y":0,"mask":mask}
	
	return img_container

	













lock=threading.Lock()

    




def index_raised(yi, y9):
	if (y9 - yi) > 40:
		return True

	return False


def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    with lock:
        if results.multi_hand_landmarks:
            for i in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
            image,
            i,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
            x8,y8=int(i.landmark[8].x*640), int(i.landmark[8].y*480)
            x12,y12=int(i.landmark[12].x*640), int(i.landmark[12].y*480)

            xi,yi= int(i.landmark[12].x*640), int(i.landmark[12].y*480)
            y9  = int(i.landmark[9].y*480)
            if index_raised(yi, y9):
                cv2.line(mask, (img_container["prev_x"],img_container["prev_y"]),(xi, yi), 0, 5)
                img_container["prev_x"],img_container["prev_y"]= xi, yi
                
            else:
                img_container["prev_x"]= xi
                img_container["prev_y"]= yi


            if (x8>x12):
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
