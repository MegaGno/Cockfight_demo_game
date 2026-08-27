import cv2
import mediapipe as mp
import numpy as np
import time

# ==========================================
# 1. ตั้งค่า MediaPipe Hand Landmarker
# ==========================================

MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = HandLandmarker.create_from_options(options)

CHICKEN_X = 800
CHICKEN_Y = 190

MOVE_SPEED = 10
# ==========================================
# 2. ตั้งค่ากล้อง
# ==========================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==========================================
# โหลดรูปไก่
# ==========================================

CHICKEN_PATH = "Gamecock_walking_outdoors_202608280048-removebg-preview.png"

chicken_img = cv2.imread(
    CHICKEN_PATH,
    cv2.IMREAD_UNCHANGED
)

if chicken_img is None:
    raise FileNotFoundError("ไม่พบรูปไก่")

# กลับด้าน เพราะต้องการให้หัวไก่อยู่ฝั่งขวา
# chicken_img = cv2.flip(chicken_img, 1)

# ปรับขนาด
CHICKEN_WIDTH = 480

ratio = CHICKEN_WIDTH / chicken_img.shape[1]

CHICKEN_HEIGHT = int(
    chicken_img.shape[0] * ratio
)

chicken_img = cv2.resize(
    chicken_img,
    (CHICKEN_WIDTH, CHICKEN_HEIGHT)
)

# ==========================================
# 3. Bounding Box ของไก่
# ==========================================

target_areas = {
    'Head': {
        'rect': (925, 250, 975, 350),
        'score': 50,
        'color': (0, 0, 255),
        'last_hit': 0
    },

    'Neck': {
        'rect': (975, 250, 1050, 350),
        'score': 40,
        'color': (0, 165, 255),
        'last_hit': 0
    },

    'Wings': {
        'rect': (850, 350, 950, 500),
        'score': 20,
        'color': (255, 0, 0),
        'last_hit': 0
    },

    'Body': {
        'rect': (950, 350, 1150, 500),
        'score': 10,
        'color': (0, 255, 0),
        'last_hit': 0
    },

    'Legs': {
        'rect': (1000, 500, 1100, 600),
        'score': 30,
        'color': (0, 255, 255),
        'last_hit': 0
    }
}

# ==========================================
# Debug Mode
# ==========================================

SHOW_HITBOX = True

# ==========================================
# 4. ระบบคะแนน
# ==========================================

total_score = 0
WIN_SCORE = 300
COOLDOWN_TIME = 1.0

floating_texts = []


# ==========================================
# 5. ฟังก์ชันตรวจสอบว่าจุดอยู่ใน Bounding Box
# ==========================================

def is_inside(px, py, rect):

    x1, y1, x2, y2 = rect

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


# ==========================================
# 6. เริ่มเกม
# ==========================================

print("==========================================")
print("       COCK CLUCK - DEMO GAME")
print("==========================================")
print("Point at the chicken with your index finger")
print("Press 'q' to quit")
print("==========================================")


start_time = time.time()

def overlay_png(background, overlay, x, y, opacity=1.0):

    overlay_h, overlay_w = overlay.shape[:2]

    # กันรูปเกินขอบจอ
    if (
        x < 0 or
        y < 0 or
        x + overlay_w > background.shape[1] or
        y + overlay_h > background.shape[0]
    ):
        return

    # RGB/BGR ของรูปไก่
    overlay_rgb = overlay[:, :, :3]

    # Alpha channel
    alpha = overlay[:, :, 3] / 255.0

    # ปรับความจาง
    alpha = alpha * opacity

    alpha = alpha[:, :, np.newaxis]

    roi = background[
        y:y + overlay_h,
        x:x + overlay_w
    ]

    blended = (
        overlay_rgb * alpha
        +
        roi * (1 - alpha)
    )

    background[
        y:y + overlay_h,
        x:x + overlay_w
    ] = blended.astype(np.uint8)


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("ไม่สามารถรับภาพจากกล้องได้")
        break


    # --------------------------------------
    # Flip ภาพเหมือนกระจก
    # --------------------------------------

    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape


    # --------------------------------------
    # ส่งภาพเข้า MediaPipe
    # --------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------
    # Timestamp ต้องเพิ่มขึ้นเรื่อย ๆ
    # --------------------------------------

    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )


    detection_result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )
    # ======================================
    # วาดไก่บนหน้าจอ
    # ======================================

    overlay_png(
        frame,
        chicken_img,
        x=CHICKEN_X,
        y=CHICKEN_Y,
        opacity=0.65
    )
    # ======================================
    # วาด Bounding Box สำหรับ Debug
    # ======================================

    if SHOW_HITBOX:

        for area_name, data in target_areas.items():

            x1, y1, x2, y2 = data['rect']
            color = data['color']

            # ถ้าเพิ่งโดน
            if time.time() - data['last_hit'] < 0.2:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    -1
                )

            else:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

            # ชื่อ Hitbox
            cv2.putText(
                frame,
                area_name,
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # ======================================
    # ตรวจจับมือ
    # ======================================

    if detection_result.hand_landmarks:

        for hand_landmarks in detection_result.hand_landmarks:

            # ----------------------------------
            # Landmark 8 = Index Finger Tip
            # ----------------------------------

            index_finger_tip = hand_landmarks[8]


            # ----------------------------------
            # แปลง normalized coordinates
            # เป็น pixel
            # ----------------------------------

            cx = int(
                index_finger_tip.x * w
            )

            cy = int(
                index_finger_tip.y * h
            )


            # ----------------------------------
            # วาดจุดปลายนิ้ว
            # ----------------------------------

            cv2.circle(
                frame,
                (cx, cy),
                15,
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.circle(
                frame,
                (cx, cy),
                20,
                (255, 255, 255),
                2
            )


            # ==================================
            # ตรวจสอบการชน
            # ==================================

            current_time = time.time()


            for area_name, data in target_areas.items():

                if is_inside(
                    cx,
                    cy,
                    data['rect']
                ):

                    # ------------------------------
                    # Cooldown
                    # ------------------------------

                    if (
                        current_time - data['last_hit']
                        >
                        COOLDOWN_TIME
                    ):

                        score_earned = data['score']


                        # เพิ่มคะแนน
                        total_score += score_earned


                        # บันทึกเวลาโดน
                        data['last_hit'] = current_time


                        print(
                            f"Hit {area_name}! "
                            f"+{score_earned} Points"
                        )


                        # ------------------------------
                        # Floating Text
                        # ------------------------------

                        floating_texts.append({

                            'text': f"+{score_earned}",

                            'pos': [
                                cx,
                                cy - 20
                            ],

                            'start_time':
                                current_time
                        })


    # ======================================
    # Floating Text
    # ======================================

    new_floating_texts = []


    for ftext in floating_texts:

        elapsed = (
            time.time()
            -
            ftext['start_time']
        )


        if elapsed < 1.0:

            curr_y = int(
                ftext['pos'][1]
                -
                (elapsed * 50)
            )


            cv2.putText(
                frame,
                ftext['text'],
                (
                    ftext['pos'][0],
                    curr_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
                cv2.LINE_AA
            )


            new_floating_texts.append(
                ftext
            )


    floating_texts = new_floating_texts


    # ======================================
    # Score UI
    # ======================================

    cv2.putText(
        frame,
        f"TOTAL SCORE: {total_score}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        4,
        cv2.LINE_AA
    )
    # ======================================
    # ตรวจสอบว่าชนะหรือยัง
    # ======================================

    if total_score >= WIN_SCORE:

        # ให้คะแนนสูงสุดเป็น 300
        total_score = WIN_SCORE

        # ทำพื้นหลังมืดลงเล็กน้อย
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (w, h),
            (0, 0, 0),
            -1
        )

        frame = cv2.addWeighted(
            overlay,
            0.5,
            frame,
            0.5,
            0
        )

        # YOU WIN
        text = "YOU WIN!"

        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            8
        )[0]

        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            8,
            cv2.LINE_AA
        )

        # แสดงคะแนนสุดท้าย
        cv2.putText(
            frame,
            f"SCORE: {total_score}",
            (text_x + 60, text_y + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )

        cv2.imshow(
            'Cock Cluck - Demo Game',
            frame
        )

        # แสดงหน้า YOU WIN 3 วินาที
        cv2.waitKey(3000)

        break

    # ======================================
    # แสดงภาพ
    # ======================================

    cv2.imshow(
        'Cock Cluck - Demo Game',
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # W = ขึ้น
    if key == ord('w'):
        CHICKEN_Y -= MOVE_SPEED

    # S = ลง
    elif key == ord('s'):
        CHICKEN_Y += MOVE_SPEED

    # A = ซ้าย
    elif key == ord('a'):
        CHICKEN_X -= MOVE_SPEED

    # D = ขวา
    elif key == ord('d'):
        CHICKEN_X += MOVE_SPEED

    # H = เปิด/ปิด Hitbox
    elif key == ord('h'):
        SHOW_HITBOX = not SHOW_HITBOX

    # Q = ออกจากเกม
    elif key == ord('q'):
        break

    print("Chicken position:", CHICKEN_X, CHICKEN_Y)

# ==========================================
# Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()

landmarker.close()