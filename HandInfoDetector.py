from cvzone.HandTrackingModule import HandDetector

# cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

def HandInfo(hands):
    # success, img = cap.read()
    # hands, img =detector.findHands(img)

    if hands:
        # Hand 1
        if len(hands) == 1:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmarks points
            bbox1 = hand1["bbox"]  # Bounding Box info x,y,w,h
            centerPoint1 = hand1["center"]  # center of the hand cx,cy
            handType1 = hand1["type"]  # Hand Type Left or Right
            # print(handType1)
            return '1'

        if len(hands) == 2:
                handright = hands[0]
                handleft = hands[1]
                lmList2 = handright["lmList"]  # List of 21 Landmarks points
                bbox2 = handright["bbox"]  # Bounding Box info x,y,w,h
                centerPoint2 = handright["center"]  # center of the hand cx,cy
                handType2 = handright["type"]  # Hand Type Left or Right

                return '2'

    # cv2.imshow("image", img)
    # cv2.waitKey(1)