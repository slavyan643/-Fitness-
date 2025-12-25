import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
import os
import threading
import time

# --- НАЛАШТУВАННЯ ---
WIN = "AI Fitness Pro V7.0 (Clean Aesthetic)"
MODEL_PATH = "voices/amy.onnx"
SCREEN_W = 1920
SCREEN_H = 1080
HIGHSCORE_FILE = "highscores.txt"

# --- НОВА ПАЛІТРА КОЛЬОРІВ (Пастель/Apple Style) ---
# BGR формат
PASTEL_CORAL = (180, 130, 240)  # М'який корал (для напруги/помилок)
PASTEL_MINT = (200, 255, 180)   # Ніжна м'ята (для успіху)
PASTEL_BLUE = (240, 220, 180)   # М'який блакитний (нейтральний)
CHARCOAL = (60, 60, 60)         # Темно-сірий для основного тексту (не чорний!)
SOFT_GREY = (160, 160, 160)     # Для другорядного тексту
WARM_WHITE = (245, 245, 245)    # Теплий білий
PURE_WHITE = (255, 255, 255)

# --- УПРАВЛІННЯ ДАНИМИ ---
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

# --- ГОЛОС ---
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

# --- НОВЕ ВИТОНЧЕНЕ МАЛЮВАННЯ ---
def draw_elegant_style(frame, landmarks, w, h, color_main):
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)
    ]
    # Дуже тонкі білі лінії
    for start_idx, end_idx in connections:
        p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        cv2.line(frame, p1, p2, PURE_WHITE, 2, cv2.LINE_AA)
    
    # Маленькі пастельні суглоби
    for idx in [11,12,13,14,15,16,23,24,25,26,27,28]:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        cv2.circle(frame, (cx, cy), 6, color_main, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, PURE_WHITE, -1, cv2.LINE_AA)

def main():
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"}, controls={"FrameRate": 30})
    picam2.configure(cfg); picam2.start()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mode = "SQUATS"
    counter = 0
    state = "UP"
    feedback_text = "READY"
    current_color = PASTEL_BLUE
    
    start_time = time.time()
    calories = 0.0
    highscores = load_highscores()
    current_highscore = highscores.get(mode, 0)
    is_new_record = False

    speak("Clean interface loaded.")

    try:
        while True:
            frame = picam2.capture_array()
            h, w, _ = frame.shape
            results = pose.process(frame)
            
            # --- ЕФЕКТ "М'ЯКОГО ФОКУСУ" ---
            # Накладаємо напівпрозорий білий шар на все відео
            white_overlay = np.full((h, w, 3), WARM_WHITE, dtype=np.uint8)
            cv2.addWeighted(white_overlay, 0.15, frame, 0.85, 0, frame)

            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            timer_text = f"{mins:02}:{secs:02}"

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                angle = 0
                
                if mode == "SQUATS":
                    p1=[lm[23].x*w,lm[23].y*h]; p2=[lm[25].x*w,lm[25].y*h]; p3=[lm[27].x*w,lm[27].y*h]
                    angle = calculate_angle(p1, p2, p3)
                    
                    if angle > 160:
                        if state == "DOWN": speak("Up")
                        state = "UP"; current_color = PASTEL_BLUE
                    if angle < 100 and state == "UP":
                        state = "DOWN"; counter += 1; calories += 0.32
                        speak(f"{counter}"); feedback_text = "PERFECT"
                        current_color = PASTEL_MINT
                        if counter > current_highscore:
                            if not is_new_record: speak("New Record!")
                            is_new_record = True; current_highscore = counter; save_highscore(mode, current_highscore)
                    if 110 < angle < 140 and state == "UP":
                        feedback_text = "LOWER"; current_color = PASTEL_CORAL

                elif mode == "PUSHUPS":
                    p1=[lm[11].x*w,lm[11].y*h]; p2=[lm[13].x*w,lm[13].y*h]; p3=[lm[15].x*w,lm[15].y*h]
                    angle = calculate_angle(p1, p2, p3)

                    if angle > 160:
                        if state == "DOWN": speak("Up")
                        state = "UP"; current_color = PASTEL_BLUE
                    if angle < 90 and state == "UP":
                        state = "DOWN"; counter += 1; calories += 0.45
                        speak(f"{counter}"); feedback_text = "STRONG"
                        current_color = PASTEL_MINT
                        if counter > current_highscore:
                            if not is_new_record: speak("New Record!")
                            is_new_record = True; current_highscore = counter; save_highscore(mode, current_highscore)
                    if 95 < angle < 130 and state == "UP":
                        feedback_text = "LOWER"; current_color = PASTEL_CORAL

                draw_elegant_style(frame, lm, w, h, current_color)

            # --- НОВИЙ MINIMALIST UI ---
            
            # 1. Блок лічильника (Верхній лівий кут, плаваючий)
            # Малюємо м'яку підкладку для тексту
            cv2.rectangle(frame, (20, 20), (350, 250), WARM_WHITE, -1)
            # Режим
            cv2.putText(frame, mode, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, SOFT_GREY, 2, cv2.LINE_AA)
            # Велика цифра (тонка і темна)
            cv2.putText(frame, str(counter), (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 4, CHARCOAL, 5, cv2.LINE_AA)
            # Reps
            cv2.putText(frame, "reps", (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, SOFT_GREY, 2, cv2.LINE_AA)
            # Рекорд
            if is_new_record:
                cv2.putText(frame, f"NEW RECORD!", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PASTEL_CORAL, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Best: {current_highscore}", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SOFT_GREY, 2, cv2.LINE_AA)

            # 2. Блок статистики (Верхній правий кут)
            cv2.rectangle(frame, (w-320, 20), (w-20, 100), WARM_WHITE, -1)
            cv2.putText(frame, f"{timer_text} | {calories:.1f} kcal", (w-300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CHARCOAL, 2, cv2.LINE_AA)

            # 3. ЦЕНТРАЛЬНИЙ СТАТУС (Найважливіше!)
            # Ніяких рамок. Просто великий, красивий кольоровий текст по центру зверху.
            status_color = PASTEL_MINT if feedback_text in ["PERFECT", "STRONG"] else (PASTEL_CORAL if feedback_text == "LOWER" else PURE_WHITE)
            if feedback_text != "READY":
                text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 3)[0]
                text_x = (w - text_size[0]) // 2
                # Малюємо легку тінь для об'єму
                cv2.putText(frame, feedback_text, (text_x+2, 202), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (200,200,200), 3, cv2.LINE_AA)
                # Основний текст
                cv2.putText(frame, feedback_text, (text_x, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.5, status_color, 3, cv2.LINE_AA)

            # Кнопки (маленькі, внизу)
            cv2.putText(frame, "1:Squats 2:Pushups R:Reset Q:Exit", (w//2 - 200, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, SOFT_GREY, 1, cv2.LINE_AA)

            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            final_frame = cv2.resize(bgr_frame, (SCREEN_W, SCREEN_H))
            cv2.imshow(WIN, final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                counter = 0; calories = 0; start_time = time.time(); feedback_text = "READY"
                is_new_record = False; speak("Reset")
            if key == ord('1'):
                mode = "SQUATS"; counter = 0; calories = 0; start_time = time.time()
                highscores = load_highscores(); current_highscore = highscores.get(mode, 0)
                is_new_record = False; speak("Squats mode")
            if key == ord('2'):
                mode = "PUSHUPS"; counter = 0; calories = 0; start_time = time.time()
                highscores = load_highscores(); current_highscore = highscores.get(mode, 0)
                is_new_record = False; speak("Pushups mode")

    except Exception as e: print(f"Error: {e}")
    finally: picam2.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
