import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 2)

while True:

    _, frame = cap.read()

    # horizontal flip
    frame = cv2.flip(frame, 1)

    # store the results in res as a rgb
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    allHands = []
    h, w, c = frame.shape

    if res.multi_hand_landmarks:
        for hand_type, handLms in zip(res.multi_handedness, res.multi_hand_landmarks):
            myHand = {}

            # get a coordinate values to create box around hands
            mylmList = []
            xList = []
            yList = []
            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                mylmList.append([px, py, pz])
                xList.append(px)
                yList.append(py)

            # box around tha hand
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2]//2), bbox[1] + (bbox[3]//2)

            myHand["lmList"] = mylmList
            myHand["bbox"] = bbox
            myHand["center"] = (cx, cy)

            # camera is flipped
            if hand_type.classification[0].label == 'Right':
                myHand["type"] = "Right"
            else:
                myHand["type"] = "Left"
            allHands.append(myHand)

            mp_drawing.draw_landmarks(frame,
                handLms,
                mp_hands.HAND_CONNECTIONS
            )
            cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20),
                (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                (255, 0, 255), 2
            )
            cv2.putText(frame, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 255), 2
            )
        
    offset = 20
    if allHands:
       
        onehand = allHands[0]
        x, y, w, h = onehand['bbox']
        frameCrop = frame[y - offset: y + h + offset, x - offset: x + w + offset]
        if frameCrop.shape[0] > 0 and frameCrop.shape[1] > 0:
            cv2.imshow("HandCrop", frameCrop)
        

    cv2.imshow("Handtracker'", frame)

    # pressing esc turns it off
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break