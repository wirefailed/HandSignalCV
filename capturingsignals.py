import cv2
import mediapipie as mp

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands = 2)

while True:

    _, frame = cap.read()

    # horizontal flip
    frame = cv2.flip(frame, 1)

    # store the results in res as a rgb
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BG2BGR))

    allHands = []
    h, w, c = frame.shape

    if results.multi_hand_landmarks:
        for hand_id, handLms in results.multi_hand_landmarks:
            myHand = {}

            # get a coordinate values to create box around hands
            mylmList = []
            xList = []
            yList = []
            for id, lm in enumerate(handLms.landmarks):
                px, py = int(lm.x * w), int(lm.y * h)
                mylmList.append([px, py, pz])
                xList.append(px)
                yList.append(py)



    # if hand is detected 
    if res.multi_hand_landmarks:
        # no for loop because we are deteching one hand for now
        mp_drawing.draw_landmarks(frame, 
            res.multi_hand_landmarks[0], 
            hands.HAND_CONNECTIONS
        )



    cv2.imshow("Handtracker'", frm)

    # pressing esc turns it off
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break