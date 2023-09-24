import cv2
import numpy as np
from keras.models import load_model
from SpatioTempralModel import loadModel
import winsound
duration = 10000  # milliseconds
freq = 1000  # Hz


# Load the trained model
model = load_model('model.h5')

# Function to calculate mean squared difference
def mean_squared(x1, x2):
    difference = x1 - x2
    a, b, c, d, e = difference.shape
    n_samples = a * b * c * d * e
    sq_diff = difference ** 2
    Sum = sq_diff.sum()
    dist = np.sqrt(Sum)
    mean_dist = dist / n_samples
    return mean_dist

# Webcam capture
#vid = r"C:\Users\HABIB\Documents\FYP\MY_test_data\02.mp4"
cap = cv2.VideoCapture(1)  # 0 corresponds to the default webcam

# Preprocessing parameters
target_shape = (227, 227)
num_frames = 10  # Batch size is 10 frames
threshold = 0.00125

frame_buffer = []

while True:
    ret, frame = cap.read()

    # Resize the frame
    resized_frame = cv2.resize(frame, target_shape)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    frame_buffer.append(gray_frame)

    if len(frame_buffer) == num_frames:
        # Stack frames to create a batch
        frame_batch = np.stack(frame_buffer, axis=-1)

        # Normalize and preprocess the batch
        normalized_batch = (frame_batch - frame_batch.mean()) / frame_batch.std()
        normalized_batch = np.clip(normalized_batch, 0, 1)

        # Reshape to match model input
        input_batch = np.expand_dims(normalized_batch, axis=(0, 4))

        # Predict using the model
        reconstructed_batch = model.predict(input_batch)

        # Calculate mean squared difference
        loss = mean_squared(input_batch, reconstructed_batch)
        print(loss)

        if loss > threshold:
            print("Anomalous batch of frames detected")
            winsound.Beep(500, 100)
            winsound.Beep(700, 100)
            

            # Here you can take further action for anomaly detection, such as raising an alarm or saving the frames
        else:
            print("Normal batch of frames")

        frame_buffer.pop(0)

    cv2.imshow('Webcam', gray_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
