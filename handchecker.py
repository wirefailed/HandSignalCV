import cv2
import mediapipe as mp

# drawing_utils visualize the results
mp_drawing = mp.solutions.drawing_utils
# change the predefined styles
mp_drawing_styles = mp.solutions.drawing_styles
# predefined recognitions for hands
mphands = mp.solutions.hands

# video captures from camera and call it 0
cap = cv2.VideoCapture(0)
# Hands() inside mphands
hands = mphands.Hands(max_num_hands=1, min_detection_confidence=0.7) # change confidence according to your favour

# to keep capturing video
while True:

    _, frame = cap.read()
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # cv2 always work with bg in default

    #storing the results
    results = hands.process(framergb) # processing the frame
    
    # post process the result 
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                handlms, 
                mphands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            ) # drawing landmarks on frames

    cv2.imshow('Handtracker', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()