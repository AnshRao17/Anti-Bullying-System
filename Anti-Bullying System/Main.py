import cv2
import mediapipe as mp
import os
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=6,
    min_detection_confidence=0.5) as hands:

    
    if not os.path.exists("Victims"):
        os.makedirs("Victims")

    
    start_time = time.time()
    selfie_count = 0

    while True:
        
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = cv2.flip(image, 1)

    
        results = hands.process(image)

        # Draw hand landmarks and bounding box
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if hand is making a "Call me" sign
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                if thumb_tip.y < index_tip.y and middle_tip.y < ring_tip.y and index_tip.y < middle_tip.y < ring_tip.y and pinky_tip.y < middle_tip.y:
                    
                    
                    selfie_count += 1
                    filename = f"Victims/selfie_{selfie_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Selfie {selfie_count} saved to {filename}!")

    
                    message = "Getting Bullied!"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    org = (100, 50)
                    fontScale = 1
                    color = (0, 255, 0)
                    thickness = 2
                    image = cv2.putText(image, message, org, font, fontScale, color, thickness, cv2.LINE_AA)

                    
                    time.sleep(1.7)

        
        cv2.imshow('Selfie Camera', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
