import cv2 
import time
import subprocess
from HandInfoDetector import HandInfo
from Utils import EmotionDetector
from cvzone.HandTrackingModule import HandDetector


############################################################


cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)



tipIds = [4, 8, 12, 16, 20] #Finger Tips Id for Mediapipe


detectorInd = HandDetector(detectionCon=0.8, maxHands=2)


link_opened = False
cTime = 0
start_time = None
tracking_duration = 60
current_emotion = None
pTime = 0
emotion_actions = {
    "happy": "xdg-open ##INSERT THE PLAYLIST LINK##",
    "sad": "xdg-open ##INSERT THE PLAYLIST LINK##",
    "angry": "xdg-open ##INSERT THE PLAYLIST LINK##",
    "neutral": "xdg-open ##INSERT THE PLAYLIST LINK##",
    "shocked": "xdg-open ##INSERT THE PLAYLIST LINK##"
}

#############################################################

while True:
    success, img = cap.read()
    emotion = EmotionDetector(img) #Emotion Detector 
    print(emotion)
    hands, img =detectorInd.findHands(img)

    if emotion in emotion_actions:
        if current_emotion != emotion:
            current_emotion = emotion
            start_time = time.time()

        elapsed_time = time.time() - start_time
        if elapsed_time >= tracking_duration and not link_opened:
            action = emotion_actions[current_emotion]
            subprocess.Popen(action, shell=True)
            link_opened = True
            current_emotion = None

    else:
        current_emotion = None
        start_time = None

    numberOfHands= HandInfo(hands)
    


    if len(hands) != 0:
    
        print(f' Number Of Hands: {numberOfHands}')           
        ### Play Pause Activation Code ###
        if numberOfHands == '1':
            fingers = []
            hand1 = hands[0]
            lmlist = hand1["lmList"]

            ### Counting the Number Displayed on Hand ###

            if lmlist[tipIds[0]][1] > lmlist[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1,5):
                if lmlist[tipIds[id]][2] < lmlist[tipIds[id]-2][2]: #Tips number's Y-Position if less than the 2nd point of the same finger, its closed and vice-versa
                    fingers.append(1)
                
                else:
                    fingers.append(0)

   
            totalFingers = fingers.count(1)
       
            print(f'Number of Fingers : {totalFingers}')

            ### Running Commands for next / previous / play / pause ###
            if totalFingers == 1:
                if fingers == [0, 0, 0, 0, 1]:
                    subprocess.Popen('playerctl --all-players next', shell=True)
                    print("Next Track Played")
            
                elif fingers == [1, 0, 0, 0, 0]:
                        subprocess.Popen('playerctl --all-players previous', shell=True)
                        print("Previous Track Played")


            elif totalFingers == 0:
                subprocess.Popen('playerctl --all-players pause', shell=True)
                print("Paused")
            
            elif totalFingers == 5:
                subprocess.Popen('playerctl --all-players play', shell=True)
                print("Play")

        ### Showing the FPS ###
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)


    cv2.imshow("IMAGE", img)
    cv2.waitKey(1)

