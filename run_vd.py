import cv2
import numpy as np

from pycoral.utils.edgetpu import make_interpreter
import tflite_runtime.interpreter as tflite

import time
import sys
import math
import os

if len(sys.argv) < 5:
    print("Usage:")
    print("python run_video.py <model> <labels> <input_video> <output_video> (cpu_thread) (confidence) (no_gen_out)")
    print("<model>: If string contain the 'edgetpu' substring then use edge tpu device else use cpu")
    print("(cpu_thread) (confidence) is optional, default is 1 thread and 0.5, cpu thread is used when we run CPU model")
    print("(no_gen_out) is optional, input `NOGEN` to stop generating output video")
    print(sys.argv)
    sys.exit(1)

# Input configuration
# ==========================================================
MODEL_PATH = sys.argv[1]
LABELS_PATH = sys.argv[2]
INPUT_VIDEO = sys.argv[3]
OUTPUT_VIDEO = sys.argv[4]
# Optional, default value
NUM_THREADS = 1
CONFIDENCE_THRESHOLD = 0.5
NO_GEN = False

if len(sys.argv) >= 6:
    NUM_THREADS = int(sys.argv[5])
if len(sys.argv) >= 7:
    CONFIDENCE_THRESHOLD = float(sys.argv[6])
if len(sys.argv) >= 8 and "NOGEN" in sys.argv[7]:
    NO_GEN = True
# ==========================================================

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

if "edgetpu" in MODEL_PATH:
    interpreter = make_interpreter(MODEL_PATH)
    print("Created Edge TPU interpreter", file=sys.stderr)
else:
    # Create interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
    print("Created interpreter", file=sys.stderr)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height = int(input_shape[1])
input_width = int(input_shape[2])

print(f"Model input size: {input_width}x{input_height}", file=sys.stderr)
print(f"Thread {NUM_THREADS} - Confidence {CONFIDENCE_THRESHOLD} - Output: {not NO_GEN}", file=sys.stderr)

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("Cannot open video.", file=sys.stderr)
    sys.exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

# Output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (frame_width, frame_height)
)

if not writer.isOpened():
    print("Cannot create writer for opencv.", file=sys.stderr)
    sys.exit(1)

print("FPS:", fps, file=sys.stderr)
print(f"Frame {frame_width} x {frame_height}:", file=sys.stderr)

frame_count = 0
inference_time_acc = 0
inf_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...", file=sys.stderr)

    frame_count += 1

    # Preprocess
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized = cv2.resize(
        rgb,
        (input_width, input_height)
    )

    input_data = np.expand_dims(
        resized.astype(np.uint8),
        axis=0
    )

    # Load data to input tensor
    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )

    # Inference
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = (time.perf_counter() - start) * 1000
    inf_idx += 1
    inference_time_acc += inference_time

    # Read outputs, only support TFver1
    boxes = interpreter.get_tensor(
        output_details[0]['index'])[0]

    classes = interpreter.get_tensor(
        output_details[1]['index'])[0]

    scores = interpreter.get_tensor(
        output_details[2]['index'])[0]

    # Draw detections
    for i in range(len(scores)):

        if scores[i] < CONFIDENCE_THRESHOLD or scores[i] > 1.0:
            continue

        ymin, xmin, ymax, xmax = boxes[i]

        left = int(xmin * frame_width)
        top = int(ymin * frame_height)
        right = int(xmax * frame_width)
        bottom = int(ymax * frame_height)
        # Check if coordinates go out of bound
        if left >= frame_width or top >= frame_height or right >= frame_width or bottom >= frame_height:
            continue

        class_id = int(classes[i])

        if class_id >= len(labels):
            label = str(class_id)
        else:
            label = labels[class_id]

        confidence = scores[i]

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"{label}: {confidence:.2f}",
            (left, max(20, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 255),
            3
        )

    # Show inference time
    cv2.putText(
        frame,
        f"Inference: {inference_time:.1f} ms",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Write output
    writer.write(frame)

    # Simulate real-time by skipping frames.
    frame_period_ms = 1000.0 / fps
    # Don't calculate the current frame, so minus 1.
    frames_to_skip = max(0, int(math.ceil(inference_time / frame_period_ms)) - 1)
    if frames_to_skip > 0:
        print(f"Skipping {frames_to_skip} frame", file=sys.stderr)
    for _ in range(frames_to_skip):
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
        # This create a freeze effect, essentially re-creating real-time camera processing.
        # By writing back the same frame, the user experiences "lag" if inference take too long.
        # finished in allowed period:     Frame1 -> Frame2 -> Frame3 -> Frame4 -> ...
        # NOT finished in allowed period: Frame1 -> Frame1 -> Frame1 -> Frame4 -> ...
        writer.write(frame)

cap.release()
writer.release()

print("Done.", file=sys.stderr)
print(f"Processed {frame_count} frames.", file=sys.stderr)

if NO_GEN and os.path.exists(OUTPUT_VIDEO):
    os.remove(OUTPUT_VIDEO)
else:
    print(f"Saved to: {OUTPUT_VIDEO}", file=sys.stderr)

# Output
inference_time_avg = inference_time_acc / inf_idx
print(f"{inference_time_avg:.1f}")
