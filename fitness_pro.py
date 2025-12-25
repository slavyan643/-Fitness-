import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
import os
import threading
import time

# --- НАЛАШТУВАННЯ ---
WIN = "AI Fitness Pro V5.0 (Neon)"
MODEL_PATH = "voices/amy.onnx"
SCREEN_W = 1920
SCREEN_H = 1080

# --- КОЛЬОРИ (BGR) ---
# Неоновий стиль
NEON_CYAN = (255, 255, 0)    # Основний колір
NEON_MAGENTA = (255, 0, 255) # Колір напруги
NEON_GREEN = (0, 255, 0)     # Успіх
WHITE = (255, 255, 255)
DARK_BG = (20, 20, 20)       # Майже чорний для панелі

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

# --- МАТЕМАТИКА ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

# --- КРАСИВЕ МАЛЮВАННЯ (НЕОН) ---
def draw_neon_style(frame, landmarks, w, h, color_main):
    # Список зв'язків (Тільки тіло, без обличчя!)
    connections = [
        (11, 12), (11, 13), (13, 15),       # Руки (ліва)
        (12, 14), (14, 16),                 # Руки (права)
        (11, 23), (12, 24), (23, 24),       # Тулуб
        (23, 25), (25, 27),                 # Ноги (ліва)
        (24, 26), (26, 28)                  # Ноги (права)
    ]

    # Малюємо лінії (Кістки)
    for start_idx, end_idx in connections:
        # Отримуємо координати
        p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
        p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
        
        # 1. Товста лінія (Світіння)
        cv2.line(frame, p1, p2, color_main, 6, cv2.LINE_AA)
        # 2. Тонка біла лінія (Центр)
        cv2.line(frame, p1, p2, WHITE, 2, cv2.LINE_AA)

    # Малюємо точки (Суглоби) - теж тільки тіло
    body_indices = [11,12,13,14,15,16,23,24,25,26,27,28]
    for idx in body_indices:
        cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
        # Зовнішнє коло
        cv2.circle(frame, (cx, cy), 8, color_main, -1, cv2.LINE_AA)
        # Внутрішнє біле
        cv2.circle(frame, (cx, cy), 3, WHITE, -1, cv2.LINE_AA)

def main():
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        controls={"FrameRate": 30}
    )
    picam2.configure(cfg)
    picam2.start()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    mode = "SQUATS"
    counter = 0
    state = "UP"
    feedback_text = "READY"
    current_color = NEON_CYAN # Початковий колір
    
    speak("Neon mode loaded.")

    try:
        while True:
            frame = picam2.capture_array()
            h, w, _ = frame.shape
            results = pose.process(frame)
            
            # --- ЕСТЕТИЧНА ПАНЕЛЬ ---
            panel_w = int(w * 0.35)
            overlay = frame.copy()
            # Темна напівпрозора панель
            cv2.rectangle(overlay, (0, 0), (panel_w, h), DARK_BG, -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                angle = 0
                main_point = (0,0)
                
                # ЛОГІКА
                if mode == "SQUATS":
                    p1 = [lm[23].x * w, lm[23].y * h]
                    p2 = [lm[25].x * w, lm[25].y * h]
                    p3 = [lm[27].x * w, lm[27].y * h]
                    angle = calculate_angle(p1, p2, p3)
                    main_point = tuple(np.multiply(p2, [1, 1]).astype(int))
                    
                    if angle > 160:
                        if state == "DOWN": speak("Up")
                        state = "UP"
                        current_color = NEON_CYAN # Розслаблений
                    
                    if angle < 100 and state == "UP":
                        state = "DOWN"
                        counter += 1
                        speak(f"Repetition {counter}")
                        feedback_text = "PERFECT"
                        current_color = NEON_GREEN # Успіх
                    
                    if 110 < angle < 140 and state == "UP":
                        feedback_text = "LOWER"
                        current_color = NEON_MAGENTA # Напруга
                        
                elif mode == "PUSHUPS":
                    p1 = [lm[11].x * w, lm[11].y * h]
                    p2 = [lm[13].x * w, lm[13].y * h]
                    p3 = [lm[15].x * w, lm[15].y * h]
                    angle = calculate_angle(p1, p2, p3)
                    main_point = tuple(np.multiply(p2, [1, 1]).astype(int))

                    if angle > 160:
                        if state == "DOWN": speak("Up")
                        state = "UP"
                        current_color = NEON_CYAN
                        
                    if angle < 90 and state == "UP":
                        state = "DOWN"
                        counter += 1
                        speak(f"Repetition {counter}")
                        feedback_text = "STRONG"
                        current_color = NEON_GREEN
                        
                    if 95 < angle < 130 and state == "UP":
                        feedback_text = "LOWER"
                        current_color = NEON_MAGENTA

                # --- МАЛЮЄМО НОВИЙ СКЕЛЕТ ---
                draw_neon_style(frame, lm, w, h, current_color)

                # Кут
                if main_point != (0,0):
                    # Гарний кружечок під текстом кута
                    cv2.circle(frame, (main_point[0]+55, main_point[1]-15), 40, DARK_BG, -1)
                    cv2.putText(frame, f"{int(angle)}", (main_point[0]+20, main_point[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, NEON_CYAN, 2, cv2.LINE_AA)

            # --- ІНТЕРФЕЙС ---
            # Заголовок
            cv2.putText(frame, mode, (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1.5, WHITE, 2, cv2.LINE_AA)
            
            # Лічильник (Великий)
            cv2.putText(frame, str(counter), (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 7, WHITE, 14, cv2.LINE_AA)
            cv2.putText(frame, "REPS", (55, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2, cv2.LINE_AA)

            # Статус бар
            bar_color = NEON_GREEN if feedback_text in ["PERFECT", "STRONG"] else (NEON_MAGENTA if feedback_text == "LOWER" else NEON_CYAN)
            cv2.rectangle(frame, (20, 450), (panel_w-20, 550), bar_color, -1)
            cv2.putText(frame, feedback_text, (40, 515), cv2.FONT_HERSHEY_SIMPLEX, 1.5, DARK_BG, 3, cv2.LINE_AA)

            # Кнопки
            cv2.putText(frame, "[1] SQUATS  [2] PUSHUPS", (30, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, NEON_CYAN, 1, cv2.LINE_AA)
            cv2.putText(frame, "[R] RESET   [Q] EXIT", (30, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,255), 1, cv2.LINE_AA)

            # Вивід
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            final_frame = cv2.resize(bgr_frame, (SCREEN_W, SCREEN_H))
            cv2.imshow(WIN, final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('r'):
                counter = 0; feedback_text = "READY"; speak("Reset")
            if key == ord('1'):
                mode = "SQUATS"; counter = 0; speak("Squats")
            if key == ord('2'):
                mode = "PUSHUPS"; counter = 0; speak("Push ups")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
