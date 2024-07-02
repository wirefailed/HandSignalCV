import cv2
import mediapipe as mp
import numpy as np
import math
import os

def createNotExistingFiles(self, letter: str) -> None:
    # check if files exist before saving datas inside the file
    if not os.path.exists(f'./Signals/training_set/{letter}'):
        os.mkdir(f'./Signals/training_set/{letter}')
    if not os.path.exists(f'./Signals/valid_set/{letter}'):
        os.mkdir(f'./Signals/valid_set/{letter}')
    if not os.path.exists(f'./Signals/test_set/{letter}'):
        os.mkdir(f'./Signals/test_set/{letter}')

def captureSignals(self, letter: str, maxData: int) -> None:
    # initialize cv2 and connect to the camera
    cap = cv2.VideoCapture(0)

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.handsl
    hands = mp_hands.Hands(max_num_hands = 1)

    # count for data sets
    maxNumOfData = maxData
    totalNum, trainSetNum, validSetNum, testSetNum = 0, 0, 0, 0

    while True:
        # read camera
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

                # box around tha hand using min and max
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2]//2), bbox[1] + (bbox[3]//2)

                # append landmarks inside myHand and other informations
                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                # label hands
                if hand_type.classification[0].label == 'Right':
                    myHand["type"] = "Right"
                else:
                    myHand["type"] = "Left"
                allHands.append(myHand)

                # draw landmarks
                mp_drawing.draw_landmarks(frame,
                    handLms,
                    mp_hands.HAND_CONNECTIONS
                )
                # draw rectangles around hand
                cv2.rectangle(frame, 
                    (bbox[0] - 20, bbox[1] - 20),
                    (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                    (255, 0, 255), 2
                )
                # input a text beside the rectangle to show which side it is
                cv2.putText(frame, 
                    myHand["type"], 
                    (bbox[0] - 30, bbox[1] - 30), 
                    cv2.FONT_HERSHEY_PLAIN, 2, 
                    (255, 0, 255), 2
                )
        
        # set a size and offset for data sets
        offset = 20
        frameSize = 300

        if allHands:
            hand = allHands[0]
            x, y, w, h = hand['bbox']
            lmList = hand['lmList']
            resized_LmList = []

            # matrix of 300 x 300 x 3(rgb) with uint8
            frameWhite = np.ones((frameSize, frameSize, 3), np.uint8) * 255

            # change the lmList depending on aspectRatio
            aspectRatio = h/w

            if aspectRatio > 1:
                k = frameSize/h
                wCalculated = math.ceil(k * w)
                wGap = math.ceil((frameSize - wCalculated)/2)

                for lm in lmList:
                    lmX = lm[0] - x + offset
                    lmY = lm[1] - y + offset
                    lmX = int(lmX * k) + wGap
                    lmY = int(lmY * k)
                    resized_LmList.append([lmX, lmY])
            else:
                k = frameSize/w
                hCalculated = math.ceil(k * h)
                hGap = math.ceil((frameSize - hCalculated)/2)

                for lm in lmList:
                    lmX = lm[0] - x + offset
                    lmY = lm[1] - y + offset
                    lmX = int(lmX * k) 
                    lmY = int(lmY * k) + hGap
                    resized_LmList.append([lmX, lmY])

            pos = resized_LmList

            # landmark conenctions based off mediapipe
            landmark_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (5, 6), (6, 7), (7, 8),
                (9, 10), (10, 11), (11, 12),
                (13, 14), (14, 15), (15, 16),
                (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17),
                (0, 5), (0, 17)
            ]

            # connect each landmarks 
            for (node1, node2) in landmark_connections:
                cv2.line(frameWhite, 
                    (pos[node1][0], pos[node1][1]), 
                    (pos[node2][0], pos[node2][1]), 
                    (0, 255, 0), 3
                )
            
            # display the landmarks in circle
            for i in range(21):
                cv2.circle(frameWhite, 
                    (pos[i][0], pos[i][1]),
                    4, (255, 0, 0), 1
                )

            cv2.imshow("frameWhite", frameWhite)

        cv2.imshow("Handtracker'", frame)

        key = cv2.waitKey(1)

        # press S to save
        if key == ord("s"):

            # randomly assign train/ valid/ test
            excludeNum = set()
            randomNum = random.choice(list(set([x for x in range(1, 3)]) - excludeNum))

            # when it gets randomly assigned into 1, it is saved into training set
            if randomNum == 1 and trainSetNum <= (maxNumOfData * 0.6):
                trainSetNum += 1
                cv2.imwrite(f'./Signals/training_set/{letter}/{totalNum}.jpg', frameWhite)
                # when it has enough data, it excludes trainset as a choice after this run
                if trainSetNum == (maxNumOfData * 0.6):
                    excludeNum.add(1)
            # when it gets randomly assigned into 2, it is saved into validation set
            elif randomNum == 2 and trainSetNum <= (maxNumOfData * 0.2):
                validSetNum += 1
                cv2.imwrite(f'./Signals/valid_set/{letter}/{totalNum}.jpg', frameWhite)
                # when it has enough data, it excludes validation set as a choice after this run
                if validSetNum == (maxNumOfData * 0.2):
                    excludeNum.add(2)
            # when it gets randomly assigned into 3, it is saved into test set
            elif randomNum == 3 and testSetNum <= (maxNumOfData * 0.2):
                testSetNum += 1
                cv2.imwrite(f'./Signals/test_set/{letter}/{totalNum}.jpg', frameWhite)
                # when it has enough data, it excludes test set as a choice after this run
                if testSetNum == (maxNumOfData*0.2):
                    excludeNum.add(3)
            # when else is reached, it means 1. error or 2. completed
            else:
                cv2.destroyAllWindows()
                cap.release()

                if totalNum == maxNumOfData:
                    print("Completed")
                else:
                    print("Failed to get enough data")
                break

            totalNum += 1

        # pressing esc turns it off
        if key == 27:
            cv2.destroyAllWindows()
            cap.release()
            break

maxNumOfDataEditable = 0 # change number maxNumofData from here

def main():
    print('Type an alphabet')
    alphabet = input()
    if alphabet >= 'a' and alphabet <= 'z':
        createNotExistingFiles(alphabet)
        captureSignals(alphabet, maxNumOfDataEditable)
    else:
        print('An Error from Main')
