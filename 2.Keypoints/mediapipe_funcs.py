import os
import cv2
import numpy as np
import mediapipe as mp

def model_init(static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5,
               model_complexity=1,
               coord_space="image"):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

    ctx = {
        "hands": hands,
        "coord_space": coord_space
    }
    return ctx

def close_model(model):
    try:
        if model and "hands" in model and model["hands"] is not None:
            model["hands"].close()
    except:
        pass

def _extract_xyzw_from_results(results, coord_space):
    """
    Devuelve (kp_42x3, lw, rw) en orden InterHand:
        Left (0 .. 20) + Right (21 .. 41)
    """
    left  = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)
    lw, rw = 0, 0

    lm_list = []
    hd_list = []

    if coord_space == "world":
        if results.multi_hand_world_landmarks:
            lm_list = results.multi_hand_world_landmarks
            hd_list = results.multi_handedness or []
    else:
        if results.multi_hand_landmarks:
            lm_list = results.multi_hand_landmarks
            hd_list = results.multi_handedness or []

    if lm_list and hd_list and len(lm_list) == len(hd_list):
        for hand_lms, handedness in zip(lm_list, hd_list):
            label = None
            try:
                label = handedness.classification[0].label  # "Left" | "Right"
            except:
                label = None
            
            pts = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(hand_lms.landmark):
                pts[i, 0] = lm.x
                pts[i, 1] = lm.y
                pts[i, 2] = lm.z

            if label and label.lower().startswith("left"):
                left = pts
                lw = 1
            elif label and label.lower().startswith("right"):
                right = pts
                rw = 1
            else:
                # fallback si no hay handeness
                if rw == 0:
                    right = pts
                    rw = 1
                else:
                    left = pts
                    lw = 1
    kp_42x3 = np.concatenate([left, right], axis=0) # 42x3
    return kp_42x3, lw, rw


def frame_process(model, frame_bgr, coord_space=None):
    """
    Procesa un frame BGR y devuelve:
    kp_42x3 (42x3), lw (0/1), rw (0/1)
    """
    if coord_space is None:
        coord_space = model.get("coord_space", "image")
    
    hands = model["hands"]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    kp_42x3, lw, rw = _extract_xyzw_from_results(results, coord_space)

    if not isinstance(kp_42x3, np.ndarray):
        kp_42x3 = np.asarray(kp_42x3, dtype=np.float32)
    
    if kp_42x3.shape != (42, 3):
        tmp = np.zeros((42, 3), dtype=np.float32)
        try:
            tmp[:kp_42x3.shape[0], :kp_42x3.shape[1]] = kp_42x3
        except:
            pass
        kp_42x3 = tmp
    kp_42x3 = kp_42x3.astype(np.float32, copy=False)
    return kp_42x3, lw, rw