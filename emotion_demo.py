import cv2
from deepface import DeepFace
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, deque
from datetime import datetime

# =============================================
# CẤU HÌNH
# =============================================

# Khởi tạo Webcam
cap = cv2.VideoCapture(0)

# Cấu hình font tiếng Việt (đảm bảo file arial.ttf nằm cùng thư mục code)
try:
    font = ImageFont.truetype("./arial.ttf", 24)
except:
    font = ImageFont.load_default()

# Từ điển dịch cảm xúc (Chương 2)
emotion_dict = {
    'angry': 'Tức giận', 'disgust': 'Ghê sợ', 'fear': 'Sợ hãi',
    'happy': 'Hạnh phúc', 'sad': 'Buồn', 'surprise': 'Bất ngờ', 'neutral': 'Trung lập'
}

# Màu cho từng cảm xúc (BGR cho OpenCV)
emotion_colors = {
    'angry': (0, 0, 255), 'disgust': (0, 128, 0), 'fear': (128, 0, 128),
    'happy': (0, 215, 255), 'sad': (255, 100, 0), 'surprise': (0, 165, 255), 'neutral': (200, 200, 200)
}

# =============================================
# DASHBOARD - Dữ liệu lịch sử
# =============================================

HISTORY_SIZE = 50          # Số frame lưu lịch sử
emotion_history = deque(maxlen=HISTORY_SIZE)   # Lịch sử cảm xúc theo frame
time_history = deque(maxlen=HISTORY_SIZE)      # Thời gian tương ứng
all_emotions_log = []                          # Toàn bộ log để tính top

# Màu cho matplotlib (RGB 0-1)
EMOTION_COLORS_PLOT = {
    'angry': '#FF4444', 'disgust': '#44AA44', 'fear': '#AA44AA',
    'happy': '#FFD700', 'sad': '#4488FF', 'surprise': '#FF8C00', 'neutral': '#AAAAAA'
}

# =============================================
# KHỞI TẠO MATPLOTLIB DASHBOARD
# =============================================

plt.ion()  # Bật chế độ interactive (không block vòng lặp)
fig = plt.figure(figsize=(12, 6), facecolor='#1a1a2e')
fig.canvas.manager.set_window_title('Dashboard - Nhận diện Cảm xúc LHU')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_bar   = fig.add_subplot(gs[0, 0])   # Biểu đồ cột phân phối cảm xúc
ax_line  = fig.add_subplot(gs[0, 1])   # Biểu đồ đường lịch sử theo thời gian
ax_pie   = fig.add_subplot(gs[1, 0])   # Biểu đồ tròn top cảm xúc
ax_info  = fig.add_subplot(gs[1, 1])   # Thông tin văn bản

# Style chung
for ax in [ax_bar, ax_line, ax_pie, ax_info]:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#0f3460')

fig.suptitle('HỆ THỐNG NHẬN DIỆN CẢM XÚC - LHU', color='#e94560',
             fontsize=13, fontweight='bold', y=0.98)

DASHBOARD_UPDATE_INTERVAL = 15  # Cập nhật dashboard mỗi N frame
frame_count = 0
current_emotion = 'neutral'
start_time = datetime.now()

print("Đang chạy hệ thống... Nhấn 'q' để thoát.")
print("Dashboard sẽ xuất hiện sau vài giây...")

# =============================================
# HÀM VẼ DASHBOARD
# =============================================

def update_dashboard():
    emotions_list = list(emotion_history)
    times_list = list(time_history)

    if not emotions_list:
        return

    # --- Xóa nội dung cũ ---
    for ax in [ax_bar, ax_line, ax_pie, ax_info]:
        ax.cla()
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#0f3460')
        ax.tick_params(colors='white', labelsize=8)

    # ---- 1. Biểu đồ cột: phân phối cảm xúc trong lịch sử gần ----
    counter = Counter(emotions_list)
    all_keys = list(emotion_dict.keys())
    counts = [counter.get(e, 0) for e in all_keys]
    labels_vi = [emotion_dict[e] for e in all_keys]
    colors = [EMOTION_COLORS_PLOT[e] for e in all_keys]

    bars = ax_bar.bar(labels_vi, counts, color=colors, edgecolor='#1a1a2e', linewidth=0.8)
    ax_bar.set_title(f'Phân phối ({HISTORY_SIZE} frame gần nhất)', color='#a8dadc', fontsize=9)
    ax_bar.set_ylabel('Số lần', color='white', fontsize=8)
    ax_bar.set_xticklabels(labels_vi, rotation=25, ha='right', fontsize=7)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', color='white', fontsize=7)

    # ---- 2. Biểu đồ đường: lịch sử cảm xúc theo thời gian ----
    emotion_labels_unique = list(emotion_dict.keys())
    emotion_numeric = [emotion_labels_unique.index(e) if e in emotion_labels_unique else 0
                       for e in emotions_list]

    if len(times_list) > 1:
        t0 = times_list[0]
        times_sec = [(t - t0).total_seconds() for t in times_list]
    else:
        times_sec = [0]

    ax_line.plot(times_sec, emotion_numeric, color='#e94560', linewidth=1.5, alpha=0.8)
    ax_line.scatter(times_sec, emotion_numeric, c=[EMOTION_COLORS_PLOT[e] for e in emotions_list],
                    s=20, zorder=5)
    ax_line.set_yticks(range(len(emotion_labels_unique)))
    ax_line.set_yticklabels([emotion_dict[e] for e in emotion_labels_unique], fontsize=7)
    ax_line.set_xlabel('Giây', color='white', fontsize=8)
    ax_line.set_title('Lịch sử cảm xúc theo thời gian', color='#a8dadc', fontsize=9)

    # ---- 3. Biểu đồ tròn: top cảm xúc toàn phiên ----
    total_counter = Counter(all_emotions_log)
    if total_counter:
        top_items = total_counter.most_common(5)
        pie_labels = [emotion_dict.get(e, e) for e, _ in top_items]
        pie_values = [v for _, v in top_items]
        pie_colors = [EMOTION_COLORS_PLOT[e] for e, _ in top_items]
        wedges, texts, autotexts = ax_pie.pie(
            pie_values, labels=pie_labels, colors=pie_colors,
            autopct='%1.0f%%', startangle=140,
            textprops={'color': 'white', 'fontsize': 7},
            wedgeprops={'edgecolor': '#1a1a2e', 'linewidth': 1}
        )
        for at in autotexts:
            at.set_fontsize(7)
        ax_pie.set_title('Top cảm xúc (toàn phiên)', color='#a8dadc', fontsize=9)

    # ---- 4. Thông tin văn bản ----
    ax_info.axis('off')
    elapsed = datetime.now() - start_time
    elapsed_str = str(elapsed).split('.')[0]
    top3 = Counter(all_emotions_log).most_common(3)
    top3_text = '\n'.join([f"  {i+1}. {emotion_dict.get(e, e)}: {v} lần"
                            for i, (e, v) in enumerate(top3)])

    info_text = (
        f"⏱  Thời gian chạy: {elapsed_str}\n\n"
        f"🕐  Giờ hiện tại:\n  {datetime.now().strftime('%H:%M:%S  %d/%m/%Y')}\n\n"
        f"😊  Cảm xúc hiện tại:\n  {emotion_dict.get(current_emotion, current_emotion)}\n\n"
        f"🏆  Top cảm xúc:\n{top3_text}\n\n"
        f"📊  Tổng frame: {frame_count}"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 color='white', fontsize=9, verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.5))
    ax_info.set_title('Thống kê', color='#a8dadc', fontsize=9)

    fig.canvas.draw()
    fig.canvas.flush_events()

# =============================================
# VÒNG LẶP CHÍNH (giữ nguyên logic gốc)
# =============================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    try:
        # Sử dụng DeepFace để phân tích (Chương 4 & 5)
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        for res in results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            emotion_en = res['dominant_emotion']
            current_emotion = emotion_en
            label = emotion_dict.get(emotion_en, emotion_en)

            # Ghi log dashboard
            emotion_history.append(emotion_en)
            time_history.append(datetime.now())
            all_emotions_log.append(emotion_en)

            # Vẽ khung và nhãn (Chương 6)
            color_bgr = emotion_colors.get(emotion_en, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

            # Hiển thị tiếng Việt bằng PIL
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            draw.text((x, y - 30), label, font=font, fill=color_bgr[::-1])  # BGR->RGB
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    except Exception:
        pass

    # Hiển thị thời gian lên frame webcam
    time_str = datetime.now().strftime('%H:%M:%S')
    cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow('He thong Nhan dien Cam xuc - LHU', frame)

    # Cập nhật dashboard định kỳ
    if frame_count % DASHBOARD_UPDATE_INTERVAL == 0:
        update_dashboard()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
print("Đã thoát hệ thống.")