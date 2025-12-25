import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
import os
import threading
import time
import gc

# --- НАЛАШТУВАННЯ ---
WIN = "AI Fitness V9.0 (Full Immersion)"
MODEL_PATH = "voices/amy.onnx"
SCREEN_W = 1920
SCREEN_H = 1080
HIGHSCORE_FILE = "highscores.txt"

# --- КОЛЬОРИ (Apple Style / Modern UI) ---
# Використовуємо яскраві, але приємні кольори для тексту
MINT = (150, 255, 150)    # Успіх
CORAL = (150, 150, 255)   # Помилка (у BGR це червоний відтінок)
WHITE = (255, 255, 255)
GLASS_BG = (30, 30, 30)   # Колір напівпрозорих плашок

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

# Функція для малювання напівпрозорих плашок (Glassmorphism)
def draw_glass_rect(frame, x, y, w, h, color, alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def main():
    picam2 = Picamera2()
    # Оптимізація: вхід 640x480 для швидкості
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
    frame_count = 0

    speak("Full screen mode ready.")

    try:
        while True:
            frame = picam2.capture_array()
            # Чистка пам'яті
            frame_count += 1
            if frame_count % 100 == 0: gc.collect()

            h, w, _ = frame.shape
            results = pose.process(frame)
            
            # --- ВІЗУАЛ: Трохи затемнюємо відео, щоб текст читався краще ---
            # Це робить вигляд "кінематографічним"
            cv2.addWeighted(frame, 0.8, np.zeros(frame.shape, frame.dtype), 0.2, 0, frame)

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

            # --- НОВИЙ ІНТЕРФЕЙС (Full Screen HUD) ---
            
            # 1. ЛІЧИЛЬНИК (Зліва зверху, великий, без фону, тільки тінь)
            # Тінь для цифри
            cv2.putText(frame, str(counter), (55, 185), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,0,0), 15, cv2.LINE_AA)
            # Сама цифра
            cv2.putText(frame, str(counter), (55, 185), cv2.FONT_HERSHEY_SIMPLEX, 6, WHITE, 5, cv2.LINE_AA)
            
            # Підпис REPS
            cv2.putText(frame, "REPS", (65, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 2, cv2.LINE_AA)
            
            # Режим тренування (над цифрою)
            cv2.putText(frame, mode, (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)

            # 2. СТАТИСТИКА (Справа зверху, акуратна плашка)
            draw_glass_rect(frame, w-350, 20, 330, 60, GLASS_BG)
            stats_text = f"{timer_text}  |  {calories:.1f} kcal"
            cv2.putText(frame, stats_text, (w-330, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
            
            # Рекорд (під статистикою)
            rec_text = "NEW RECORD!" if is_new_record else f"Best: {current_highscore}"
            rec_color = MINT if is_new_record else (200, 200, 200)
            cv2.putText(frame, rec_text, (w-330, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 1, cv2.LINE_AA)

            # 3. ЦЕНТРАЛЬНИЙ СТАТУС (Величезний по центру)
            # Колір залежить від ситуації
            if biomech_error: status_color = CORAL
            elif feedback_text in ["PERFECT", "STRONG"]: status_color = MINT
            elif state is None: status_color = (255, 200, 100) # Жовтуватий для очікування
            else: status_color = WHITE

            if feedback_text != "READY":
                text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
                text_x = (w - text_size[0]) // 2
                text_y = h // 2  # По центру екрана
                
                # Малюємо темну "підкладку" (тінь), щоб читалося на будь-якому фоні
                cv2.putText(frame, feedback_text, (text_x+5, text_y+5), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), 8, cv2.LINE_AA)
                cv2.putText(frame, feedback_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 3, status_color, 5, cv2.LINE_AA)

            # 4. ПІДКАЗКИ (Знизу по центру, маленькі)
            keys_text = "[1] SQUATS   [2] PUSHUPS   [R] RESET"
            keys_size = cv2.getTextSize(keys_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(frame, keys_text, ((w-keys_size[0])//2, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

            # Розтягуємо на 1920x1080
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
