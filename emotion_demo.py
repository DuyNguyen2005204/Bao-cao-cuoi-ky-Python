import cv2
from deepface import DeepFace
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Khởi tạo Webcam
cap = cv2.VideoCapture(0)

# Cấu hình font tiếng Việt (đảm bảo file arial.ttf nằm cùng thư mục code)
try:
    font = ImageFont.truetype("./arial.ttf", 24)
except:
    font = ImageFont.load_default()

# Từ điển dịch cảm xúc (Chương 2)
emotion_dict = {
    'angry': 'Gian du', 'disgust': 'Ghe so', 'fear': 'So hai',
    'happy': 'Hanh phuc', 'sad': 'Buon', 'surprise': 'Bat ngo', 'neutral': 'Trung lap'
}

print("Đang chạy hệ thống... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret: break

    try:
        # Sử dụng DeepFace để phân tích (Chương 4 & 5)
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        for res in results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            emotion_en = res['dominant_emotion']
            label = emotion_dict.get(emotion_en, emotion_en)
            
            # Vẽ khung và nhãn (Chương 6)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Hiển thị tiếng Việt bằng PIL
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y - 30), label, font=font, fill=(0, 255, 0))
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
    except Exception:
        pass

    cv2.imshow('He thong Nhan dien Cam xuc - LHU', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()