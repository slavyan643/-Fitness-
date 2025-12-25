import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
import os
import threading
import time
import gc # Додали збирач сміття для очистки пам'яті

# --- НАЛАШТУВАННЯ ---
WIN = "AI Fitness Pro V8.2 (Stable Lite)"
MODEL_PATH = "voices/amy.onnx"
SCREEN_W = 1920
SCREEN_H = 1080
HIGHSCORE_FILE = "highscores.txt"

# --- КОЛЬОРИ ---
PASTEL_CORAL = (180, 130, 240)
PASTEL_MINT = (200, 255, 180)
PASTEL_BLUE = (240, 220, 180)
CHARCOAL = (50, 50, 50)
SOFT_GREY = (160, 160, 160)
WARM_WHITE = (245, 245, 245)
PURE_WHITE = (255, 255, 255)

def load_highscores():
    scores = {"SQUATS": 0, "PUSHUPS": 0}
    if os.path.exists(HIGHSCORE_FILE):
        with open(HIGHSCORE_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2: scores[parts[0]] = int(parts[1])
    return scores

def save_highscore(mode, score):
    scores = load_highscores()
    if score > scores.get(mode, 0):
        scores[mode] = score
        with open(HIGHSCORE_FILE, "w") as f:
            for k, v in scores.items(): f.write(f"{k}:{v}\n")
        return True
    return False

last_speech_time = 0
def speak_worker(text):
    safe_text = ". " + text
    cmd = f'echo "{safe_text}" | piper --model {MODEL_PATH} --output_raw | aplay -r 22050 -f S16_LE -t raw - 2>/dev/null'
    os.system(cmd)

def speak(text, cooldown=1.5):
    global last_speech_time
    current_time = time.time()
    if current_time - last_speech_time > cooldown:
        last_speech_time = current_time
        threading.Thread(target=speak_worker, args=(text,)).start()

def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def main():
    picam2 = Picamera2()
    # ОПТИМІЗАЦІЯ: Зменшуємо вхідну картинку до 640x480 і FPS до 20
    # MediaPipe все одно стискає картинку, тому HD якість на вході тільки гріє процесор дарма.
    cfg = picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"},
        controls={"FrameRate": 20}
    )
    picam2.configure(cfg); picam2.start()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mode = "SQUATS"
    counter = 0
    state = None 
    feedback_text = "STAND UP"
    
    start_time = time.time()
    calories = 0.0
    highscores = load_highscores()
    current_highscore = highscores.get(mode, 0)
    is_new_record = False
    biomech_error = False 
    
    # Лічильник кадрів для очистки пам'яті
    frame_count = 0

    speak("Optimized mode ready.")

    try:
        while True:
            frame = picam2.capture_array()
            # ОПТИМІЗАЦІЯ: Примусово чистимо пам'ять кожні 100 кадрів
            frame_count += 1
            if frame_count % 100 == 0:
                gc.collect()

            h, w, _ = frame.shape
            results = pose.process(frame)
            
            # Ефект м'якого фокусу (легша версія)
            white_overlay = np.full((h, w, 3), WARM_WHITE, dtype=np.uint8)
            cv2.addWeighted(white_overlay, 0.1, frame, 0.9, 0, frame)

            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            timer_text = f"{mins:02}:{secs:02}"
            
            biomech_error = False

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                angle = 0
                
                if mode == "SQUATS":
                    p1=[lm[23].x*w,lm[23].y*h]; p2=[lm[25].x*w,lm[25].y*h]; p3=[lm[27].x*w,lm[27].y*h]
                    angle = calculate_angle(p1, p2, p3)
                    
                    knee_dist = abs(lm[25].x - lm[26].x)
                    ankle_dist = abs(lm[27].x - lm[28].x)
                    
                    if state == "DOWN" and knee_dist < (ankle_dist * 0.7):
                        feedback_text = "KNEES OUT!"; biomech_error = True

                    if angle > 165: 
                        if state is None: speak("Ready"); feedback_text = "READY"
                        if state == "DOWN": speak("Up")
                        state = "UP"
                    
                    if angle < 100 and state == "UP":
                        state = "DOWN"
                        if not biomech_error:
                            counter += 1; calories += 0.32; speak(str(counter))
                            feedback_text = "PERFECT"
                            if counter > current_highscore:
                                if not is_new_record: speak("New Record!")
                                is_new_record = True; current_highscore = counter; save_highscore(mode, current_highscore)
                    
                    if 110 < angle < 140 and state == "UP" and not biomech_error:
                        feedback_text = "LOWER"

                elif mode == "PUSHUPS":
                    p1=[lm[11].x*w,lm[11].y*h]; p2=[lm[13].x*w,lm[13].y*h]; p3=[lm[15].x*w,lm[15].y*h]
                    angle = calculate_angle(p1, p2, p3)

                    sh = [lm[11].x*w,lm[11].y*h]; hip = [lm[23].x*w,lm[23].y*h]; ank = [lm[27].x*w,lm[27].y*h]
                    if calculate_angle(sh, hip, ank) < 160:
                        feedback_text = "FIX BACK"; biomech_error = True

                    if angle > 165:
                        if state is None: speak("Ready"); feedback_text = "READY"
                        if state == "DOWN": speak("Up")
                        state = "UP"
                    
                    if angle < 90 and state == "UP":
                        state = "DOWN"
                        if not biomech_error:
                            counter += 1; calories += 0.45; speak(str(counter))
                            feedback_text = "STRONG"
                            if counter > current_highscore:
                                if not is_new_record: speak("New Record!")
                                is_new_record = True; current_highscore = counter; save_highscore(mode, current_highscore)
                    
                    if 95 < angle < 130 and state == "UP" and not biomech_error:
                        feedback_text = "LOWER"

            # --- UI ---
            cv2.rectangle(frame, (20, 20), (320, 100), WARM_WHITE, -1)
            cv2.putText(frame, f"{mode} | {counter}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CHARCOAL, 2, cv2.LINE_AA)

            cv2.rectangle(frame, (w-320, 20), (w-20, 100), WARM_WHITE, -1)
            cv2.putText(frame, f"{timer_text} | {calories:.1f} kcal", (w-300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CHARCOAL, 2, cv2.LINE_AA)

            if biomech_error: status_color = PASTEL_CORAL
            elif feedback_text in ["PERFECT", "STRONG"]: status_color = PASTEL_MINT
            elif state is None: status_color = PASTEL_BLUE
            else: status_color = PURE_WHITE

            if feedback_text != "READY":
                text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 3)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, feedback_text, (text_x+2, 202), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (200,200,200), 3, cv2.LINE_AA)
                cv2.putText(frame, feedback_text, (text_x, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, status_color, 3, cv2.LINE_AA)

            if is_new_record:
                cv2.putText(frame, "NEW RECORD!", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PASTEL_CORAL, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Best: {current_highscore}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, SOFT_GREY, 1, cv2.LINE_AA)

            cv2.putText(frame, "Keys: [1] Squats [2] Pushups [R] Reset", (w//2 - 200, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, SOFT_GREY, 1, cv2.LINE_AA)

            # Розтягуємо маленьку картинку на великий екран (це швидко для GPU)
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            final_frame = cv2.resize(bgr_frame, (SCREEN_W, SCREEN_H))
            cv2.imshow(WIN, final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                counter = 0; calories = 0; start_time = time.time(); feedback_text = "STAND UP"; state = None
                is_new_record = False; speak("Reset")
            if key == ord('1'):
                mode = "SQUATS"; counter = 0; calories = 0; start_time = time.time(); state = None
                highscores = load_highscores(); current_highscore = highscores.get(mode, 0)
                is_new_record = False; speak("Squats")
            if key == ord('2'):
                mode = "PUSHUPS"; counter = 0; calories = 0; start_time = time.time(); state = None
                highscores = load_highscores(); current_highscore = highscores.get(mode, 0)
                is_new_record = False; speak("Pushups")

    except Exception as e: print(f"Error: {e}")
    finally: picam2.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
