import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import time
import streamlit as st 




def index_raised(yi, y9):
    if(y9-yi)>40:
        return True
    return False




# drawing tools


mask = np.ones((480, 640))*255
mask = mask.astype('uint8')








draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)



RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

global max_x, max_y 
global ml 
global curr_tool
global time_init
global rad
global thick
global prevx, prevy

ml = 150
max_x, max_y = 250+ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0,0
def getTool(x):
    if x < 50 + ml:
        return "line"
    elif x<100 + ml:
        return "rectangle"
    elif x < 150 + ml:
        return"draw"
    elif x<200 + ml:
        return "circle"
    else:
        return "erase"

class VideoProcessor:
    mask = np.ones((480, 640))*255
    mask = mask.astype('uint8')
    def getTool(x):
        if x < 50 + ml:
            return "line"
        elif x<100 + ml:
            return "rectangle"
        elif x < 150 + ml:
            return"draw"
        elif x<200 + ml:
            return "circle"
        else:
            return "erase"

    def index_raised(yi, y9):
        print("raise")
        if(y9-yi)>40:
            return True
        return False








    





    def recv(self, frame):
        global xii,yii
        global max_x, max_y 
        global ml 
        global curr_tool
        global time_init
        global rad
        global thick
        global prevx, prevy
        global var_inits
        frm = frame.to_ndarray(format="bgr24")

        # img = process(img)
        op = hands.process(frm)
        if op.multi_hand_landmarks:
            for i in op.multi_hand_landmarks:
                draw.draw_landmarks(
                frm,i,
            mp_hands.HAND_CONNECTIONS)
            x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)
                


        cv2.putText(frm, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)




        return av.VideoFrame.from_ndarray(frm, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
