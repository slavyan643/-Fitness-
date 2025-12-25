import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
import os
import threading
import time

# --- НАЛАШТУВАННЯ ---
WIN = "AI Fitness Pro V4"
MODEL_PATH = "voices/amy.onnx"
SCREEN_W = 1920
SCREEN_H = 1080

# Кольори (BGR)
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
DARK = (30, 30, 30)

# --- МОДУЛЬ ГОЛОСУ ---
last_speech_time = 0
def speak_worker(text):
    # Використовуємо piper для швидкого синтезу
    cmd = f'echo "{text}" | piper --model {MODEL_PATH} --output_raw | aplay -r 22050 -f S16_LE -t raw - 2>/dev/null'
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

def main():
    # 1. ЗАПУСК КАМЕРИ
    picam2 = Picamera2()
    # Захоплюємо в 720p для швидкості, потім розтягнемо
    cfg = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        controls={"FrameRate": 30}
    )
    picam2.configure(cfg)
    picam2.start()

    # 2. MEDIAPIPE
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    draw_spec_p = mp_drawing.DrawingSpec(color=CYAN, thickness=5, circle_radius=5)
    draw_spec_l = mp_drawing.DrawingSpec(color=(255,255,255), thickness=3)
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 3. ВІКНО НА ВЕСЬ ЕКРАН
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 4. ЗМІННІ СТАНУ
    mode = "SQUATS" # Режими: "SQUATS" або "PUSHUPS"
    counter = 0
    state = "UP"
    feedback_text = "READY"
    
    speak("System ready. Select mode.")

    try:
        while True:
            frame = picam2.capture_array()
            h, w, _ = frame.shape
            results = pose.process(frame)
            
            # Малюємо темну панель зліва
            panel_w = int(w * 0.35)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (panel_w, h), DARK, -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                angle = 0
                main_point = (0,0)
                
                # --- ЛОГІКА: ПРИСІДАННЯ (SQUATS) ---
                if mode == "SQUATS":
                    # Точки: Стегно(23) - Коліно(25) - Щиколотка(27)
                    p1 = [lm[23].x * w, lm[23].y * h]
                    p2 = [lm[25].x * w, lm[25].y * h]
                    p3 = [lm[27].x * w, lm[27].y * h]
                    angle = calculate_angle(p1, p2, p3)
                    main_point = tuple(np.multiply(p2, [1, 1]).astype(int))
                    
                    if angle > 160:
                        if state == "DOWN": speak("Up")
                        state = "UP"
                    if angle < 100 and state == "UP":
                        state = "DOWN"
                        counter += 1
                        speak(str(counter))
                        feedback_text = "GOOD"
                    if 110 < angle < 140 and state == "UP":
                        feedback_text = "LOWER"
                        cv2.line(frame, tuple(map(int, p1)), tuple(map(int, p2)), RED, 5)

                # --- ЛОГІКА: ВІДЖИМАННЯ (PUSHUPS) ---
                elif mode == "PUSHUPS":
                    # Точки: Плече(11) - Лікоть(13) - Зап'ястя(15)
                    p1 = [lm[11].x * w, lm[11].y * h]
                    p2 = [lm[13].x * w, lm[13].y * h]
                    p3 = [lm[15].x * w, lm[15].y * h]
                    angle = calculate_angle(p1, p2, p3)
                    main_point = tuple(np.multiply(p2, [1, 1]).astype(int))

                    if angle > 160:
                        if state == "DOWN": speak("Up")
                        state = "UP"
                    if angle < 90 and state == "UP":
                        state = "DOWN"
                        counter += 1
                        speak(str(counter))
                        feedback_text = "STRONG"
                    if 95 < angle < 130 and state == "UP":
                        feedback_text = "LOWER"

                # Виводимо кут біля суглоба
                if main_point != (0,0):
                    cv2.putText(frame, f"{int(angle)}", (main_point[0]+20, main_point[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, CYAN, 2)
                
                # Малюємо скелет
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, draw_spec_p, draw_spec_l)

            # --- ІНТЕРФЕЙС ---
            # Назва режиму
            cv2.putText(frame, mode, (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (200,200,200), 2)
            # Лічильник
            cv2.putText(frame, str(counter), (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, (255,255,255), 12)
            cv2.putText(frame, "reps", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,150,150), 2)

            # Блок статусу
            color = GREEN if feedback_text in ["GOOD", "STRONG"] else (ORANGE if feedback_text == "LOWER" else CYAN)
            cv2.rectangle(frame, (20, 450), (panel_w-20, 550), color, -1)
            cv2.putText(frame, feedback_text, (40, 515), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 3)

            # Підказки кнопок
            cv2.putText(frame, "[1] SQUATS  [2] PUSHUPS", (30, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 1)
            cv2.putText(frame, "[R] RESET   [Q] EXIT", (30, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)

            # Розтягування на весь екран (1920x1080)
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            final_frame = cv2.resize(bgr_frame, (SCREEN_W, SCREEN_H))
            cv2.imshow(WIN, final_frame)

            # Обробка клавіш
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
