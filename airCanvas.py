import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import datetime

# -------------------------------
# Mediapipe Setup
# -------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# -------------------------------
# Canvas Setup
# -------------------------------
bpoints, gpoints, rpoints, ypoints = [deque(maxlen=1024) for _ in range(4)]
colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255)]
color_names = ["BLUE","GREEN","RED","YELLOW"]
colorIndex = 0

# Bigger paint canvas
paintWindow = np.ones((600, 1280, 3), dtype=np.uint8) * 255

# -------------------------------
# Toolbar Buttons (spread evenly)
# -------------------------------
buttons = {
    "CLEAR": (50, 1, 160, 65),
    "SAVE": (180, 1, 290, 65),
    "BLUE": (310, 1, 405, 65),
    "GREEN": (425, 1, 520, 65),
    "RED": (540, 1, 635, 65),
    "YELLOW": (655, 1, 780, 65),   # Extended width
    "EXIT": (800, 1, 910, 65)
}

def draw_toolbar(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (img.shape[1],70), (45,45,45), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    for name,(x1,y1,x2,y2) in buttons.items():
        if name in ["CLEAR","SAVE","EXIT"]: color = (230,230,230)
        elif name == "BLUE": color = (255,0,0)
        elif name == "GREEN": color = (0,255,0)
        elif name == "RED": color = (0,0,255)
        elif name == "YELLOW": color = (0,255,255)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,-1)
        cv2.putText(img,name,(x1+10,40),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

def draw_status_panel(frame, mode, colorName, fingers_up):
    h, w, _ = frame.shape
    status = f"Mode: {'DRAWING' if mode else 'PAUSED'}"
    col_text = f"Color: {colorName}"
    fingers_text = f"Fingers: {fingers_up}"
    mode_color = (0,255,0) if mode else (0,0,255)
    cv2.rectangle(frame,(10,h-70),(320,h-10),(240,240,240),-1)
    cv2.putText(frame,status,(20,h-50),cv2.FONT_HERSHEY_SIMPLEX,0.6,mode_color,2)
    cv2.putText(frame,col_text,(20,h-25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(80,80,80),2)
    cv2.putText(frame,fingers_text,(190,h-25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,100,100),1)

def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = 0
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
            fingers += 1
    return fingers

# -------------------------------
# Camera Setup
# -------------------------------
cap = cv2.VideoCapture(0)
# Force wide camera frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_center = None
drawing_mode = False
just_resumed = False

# -------------------------------
# Main Loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    h, w, _ = frame.shape

    draw_toolbar(frame)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    center=None
    fingers_up=0

    # Hand Detection
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            cx=int(hand_landmarks.landmark[8].x*w)
            cy=int(hand_landmarks.landmark[8].y*h)
            center=(cx,cy)
            cv2.circle(frame,center,8,(0,255,255),-1)
            fingers_up=count_fingers(hand_landmarks)

            # 1 finger → draw; >1 → pause
            if fingers_up == 1:
                if not drawing_mode:
                    drawing_mode=True
                    prev_center=None
                    just_resumed=True
                    print("✏️ One finger detected → Drawing started")
            else:
                if drawing_mode:
                    drawing_mode=False
                    prev_center=None
                    print("✋ Multiple fingers → Drawing paused")

    draw_status_panel(frame,drawing_mode,color_names[colorIndex],fingers_up)

    # Toolbar interaction
    if center and center[1] <= 65:
        for name,(x1,y1,x2,y2) in buttons.items():
            if x1 <= center[0] <= x2:
                if name=="CLEAR":
                    paintWindow[:] = 255
                    bpoints,gpoints,rpoints,ypoints = [deque(maxlen=1024) for _ in range(4)]
                    print("🧹 Canvas cleared")
                elif name=="SAVE":
                    filename=f"drawing_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                    cv2.imwrite(filename,paintWindow)
                    print(f"💾 Saved as {filename}")
                elif name=="EXIT":
                    print("👋 Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                elif name in color_names:
                    colorIndex=color_names.index(name)
                    print(f"🎨 Color changed to {name}")
        just_resumed=True
        continue

    # Drawing logic
    if center and drawing_mode:
        if just_resumed:
            prev_center=None
            just_resumed=False
        if prev_center:
            cx=int(0.4*center[0]+0.6*prev_center[0])
            cy=int(0.4*center[1]+0.6*prev_center[1])
            center=(cx,cy)
        prev_center=center
        points_list=[bpoints,gpoints,rpoints,ypoints][colorIndex]
        if len(points_list)==0:
            points_list.append(deque(maxlen=1024))
        points_list[-1].appendleft(center)
    else:
        [bpoints,gpoints,rpoints,ypoints][colorIndex].append(deque(maxlen=1024))

    # Draw lines
    for i,points in enumerate([bpoints,gpoints,rpoints,ypoints]):
        for j in range(len(points)):
            for k in range(1,len(points[j])):
                if points[j][k-1] is None or points[j][k] is None: continue
                cv2.line(frame,points[j][k-1],points[j][k],colors[i],4)
                cv2.line(paintWindow,points[j][k-1],points[j][k],colors[i],4)

    cv2.imshow("Air Canvas",frame)
    cv2.imshow("Paint",paintWindow)

    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
