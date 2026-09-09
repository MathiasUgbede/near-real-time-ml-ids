import socket
import json
import time
import warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Path to the previously trained Random Forest model
MODEL_PATH = '/home/mathias/ml-ids-project/models/random_forest_model.pkl'

# Folder where experiment results will be saved
RESULTS_PATH = '/home/mathias/ml-ids-project/results/'

# VM2 listens on all available network interfaces
HOST = '0.0.0.0'

# TCP port used for communication between VM1 and VM2
PORT = 9999

# First 10 flows are excluded from LATENCY statistics
# to reduce initial model/system warm-up effects
WARMUP = 10


print("=" * 60)
print("IDS DETECTOR (VM2) - REAL-TIME CLASSIFICATION")
print("=" * 60)
print("")


# ============================================================
# 2. LOAD THE TRAINED RANDOM FOREST MODEL
# ============================================================

print("[1] Loading Random Forest model...")

# Loads the saved trained model.
# The model is NOT retrained during the streamed experiment.
model = joblib.load(MODEL_PATH)

print("    Model loaded successfully")


# ============================================================
# 3. CREATE TCP SERVER ON VM2
# ============================================================

print("")
print("[2] Starting detector on port " + str(PORT) + "...")

# AF_INET = IPv4
# SOCK_STREAM = TCP
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Allows the socket address to be reused after restarting the script
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind VM2 detector to port 9999
server_socket.bind((HOST, PORT))

# Listen for one traffic-generator connection
server_socket.listen(1)

print("    Waiting for connection from VM1...")


# Wait until VM1 connects to the detector
conn, addr = server_socket.accept()

print("    Connected to traffic generator at " + str(addr[0]))


# ============================================================
# 4. INITIALISE STORAGE FOR EXPERIMENT RESULTS
# ============================================================

print("")
print("[3] Receiving and classifying flows...")

# Stores pure Random Forest prediction times
classify_times = []

# Stores detector-side processing times:
# record arrival on VM2 -> classification decision
total_times = []

# Store model predictions
predictions = []

# Store actual labels (ground truth) for accuracy calculation
actuals = []

flow_count = 0

# TCP is stream-based, so a buffer is used to reconstruct
# complete newline-separated JSON records
buffer = ""

# Start timing the complete detector session
start_session = time.time()


# ============================================================
# 5. RECEIVE RECORDS FROM VM1 AND CLASSIFY THEM
# ============================================================

try:
    while True:

        # Receive data from VM1 and convert bytes to text
        data = conn.recv(4096).decode('utf-8')

        # Stop if VM1 closes the connection
        if not data:
            break

        # Add received TCP data to the buffer
        buffer += data

        # Process each complete newline-separated record
        while "\n" in buffer:

            line, buffer = buffer.split("\n", 1)

            # VM1 sends END when all records have been transmitted
            if line.strip() == "END":
                raise StopIteration

            # Ignore empty lines
            if not line.strip():
                continue


            # ------------------------------------------------
            # RECORD ARRIVES AT VM2
            # ------------------------------------------------

            # Detector-side timing starts AFTER the record
            # has arrived at VM2
            arrival_time = time.time()


            # ------------------------------------------------
            # PREPARE RECEIVED RECORD
            # ------------------------------------------------

            # Convert JSON string back into Python data
            record = json.loads(line)

            # Extract feature values and reshape them as
            # one sample in the format expected by the model
            features = np.array(
                record['features']
            ).reshape(1, -1)

            # Ground-truth label used ONLY for evaluation.
            # It is NOT supplied to the model for prediction.
            actual_label = record['label']


            # ------------------------------------------------
            # RANDOM FOREST CLASSIFICATION
            # ------------------------------------------------

            # Start pure model-prediction timer
            classify_start = time.time()

            # Random Forest predicts the binary class:
            # 0 = Normal
            # 1 = Malicious
            prediction = model.predict(features)[0]

            # Stop model-prediction timer
            classify_end = time.time()


            # ------------------------------------------------
            # CALCULATE LATENCY
            # ------------------------------------------------

            # Pure Random Forest prediction time
            classify_ms = (
                classify_end - classify_start
            ) * 1000

            # Detector-side processing latency:
            # arrival on VM2 -> classification decision
            total_ms = (
                classify_end - arrival_time
            ) * 1000


            # ------------------------------------------------
            # STORE RESULTS
            # ------------------------------------------------

            classify_times.append(classify_ms)
            total_times.append(total_ms)

            predictions.append(int(prediction))
            actuals.append(int(actual_label))

            flow_count += 1

            # Display progress after every 100 flows
            if flow_count % 100 == 0:
                print(
                    "    Processed "
                    + str(flow_count)
                    + " flows..."
                )


# END signal is used to exit the nested receive loops
except StopIteration:
    print("    Received END signal from VM1")


# ============================================================
# 6. CLOSE CONNECTION AND CALCULATE SESSION TIME
# ============================================================

session_time = time.time() - start_session

conn.close()
server_socket.close()


# ============================================================
# 7. REMOVE WARM-UP FLOWS FROM LATENCY STATISTICS
# ============================================================

# Remove the first 10 observations from latency analysis
classify_arr = np.array(classify_times[WARMUP:])
total_arr = np.array(total_times[WARMUP:])

# 1,000 streamed - 10 warm-up = 990 latency observations
kept = len(classify_arr)


# ============================================================
# 8. PURE MODEL CLASSIFICATION LATENCY
# ============================================================

print("")
print("[4] Classification Latency (pure model prediction):")

print(
    "    Flows in statistics: "
    + str(kept)
    + " (first "
    + str(WARMUP)
    + " discarded as warm-up)"
)

print(
    "    Average: "
    + format(np.mean(classify_arr), '.4f')
    + " ms"
)

print(
    "    Median:  "
    + format(np.median(classify_arr), '.4f')
    + " ms"
)

print(
    "    Min:     "
    + format(np.min(classify_arr), '.4f')
    + " ms"
)

print(
    "    Max:     "
    + format(np.max(classify_arr), '.4f')
    + " ms"
)

print(
    "    Std dev: "
    + format(np.std(classify_arr), '.4f')
    + " ms"
)


# ============================================================
# 9. DETECTOR-SIDE PROCESSING LATENCY
# ============================================================

print("")

# IMPORTANT:
# This is detector-side latency, NOT full network end-to-end
# latency, because timing begins after the record arrives at VM2.
print("[5] Detector-Side Processing Latency (arrival to decision):")

print(
    "    Average: "
    + format(np.mean(total_arr), '.4f')
    + " ms"
)

print(
    "    Median:  "
    + format(np.median(total_arr), '.4f')
    + " ms"
)

print(
    "    Min:     "
    + format(np.min(total_arr), '.4f')
    + " ms"
)

print(
    "    Max:     "
    + format(np.max(total_arr), '.4f')
    + " ms"
)

print(
    "    Std dev: "
    + format(np.std(total_arr), '.4f')
    + " ms"
)


# ============================================================
# 10. CALCULATE THROUGHPUT
# ============================================================

# Throughput = total flows processed / total session time
# This represents the throughput of this sequential,
# single-threaded experimental setup.
throughput = (
    flow_count / session_time
    if session_time > 0
    else 0
)

print("")

print(
    "    Total session time: "
    + format(session_time, '.2f')
    + " seconds"
)

print(
    "    Throughput: "
    + format(throughput, '.1f')
    + " flows per second"
)


# ============================================================
# 11. CALCULATE STREAMED CLASSIFICATION ACCURACY
# ============================================================

# Compare each model prediction with its actual ground-truth label
correct = sum(
    1
    for p, a in zip(predictions, actuals)
    if p == a
)

# Accuracy uses ALL streamed flows.
# The 10-flow warm-up exclusion applies only to latency statistics.
accuracy = (
    correct / flow_count
    if flow_count > 0
    else 0
)

print("")

print(
    "[6] Detection accuracy on streamed flows: "
    + format(accuracy * 100, '.2f')
    + "%"
)


# ============================================================
# 12. SAVE EXPERIMENT RESULTS
# ============================================================

print("")
print("[7] Saving results...")

results_file = (
    RESULTS_PATH
    + 'realtime_detection_results.txt'
)

with open(results_file, 'w') as f:

    f.write(
        "TWO-VM REAL-TIME DETECTION RESULTS "
        "(Random Forest)\n"
    )

    f.write("=" * 50 + "\n")

    f.write(
        "Flows processed (total): "
        + str(flow_count)
        + "\n"
    )

    f.write(
        "Flows in statistics: "
        + str(kept)
        + " (warm-up "
        + str(WARMUP)
        + " discarded)\n"
    )


    # --------------------------------------------------------
    # Save pure model prediction latency
    # --------------------------------------------------------

    f.write(
        "\n-- Classification latency (pure model) --\n"
    )

    f.write(
        "Average: "
        + format(np.mean(classify_arr), '.4f')
        + " ms\n"
    )

    f.write(
        "Median:  "
        + format(np.median(classify_arr), '.4f')
        + " ms\n"
    )

    f.write(
        "Min:     "
        + format(np.min(classify_arr), '.4f')
        + " ms\n"
    )

    f.write(
        "Max:     "
        + format(np.max(classify_arr), '.4f')
        + " ms\n"
    )

    f.write(
        "Std dev: "
        + format(np.std(classify_arr), '.4f')
        + " ms\n"
    )


    # --------------------------------------------------------
    # Save detector-side processing latency
    # --------------------------------------------------------

    f.write(
        "\n-- Detector-Side Processing Latency "
        "(arrival to decision) --\n"
    )

    f.write(
        "Average: "
        + format(np.mean(total_arr), '.4f')
        + " ms\n"
    )

    f.write(
        "Median:  "
        + format(np.median(total_arr), '.4f')
        + " ms\n"
    )

    f.write(
        "Min:     "
        + format(np.min(total_arr), '.4f')
        + " ms\n"
    )

    f.write(
        "Max:     "
        + format(np.max(total_arr), '.4f')
        + " ms\n"
    )

    f.write(
        "Std dev: "
        + format(np.std(total_arr), '.4f')
        + " ms\n"
    )

    f.write(
        "\nThroughput: "
        + format(throughput, '.1f')
        + " flows per second\n"
    )

    f.write(
        "Detection accuracy: "
        + format(accuracy * 100, '.2f')
        + "%\n"
    )


print("    Saved: " + results_file)


# ============================================================
# 13. SAVE LATENCY ARRAYS FOR VISUALISATION
# ============================================================

# Save NumPy arrays so they can later be used
# by the latency analysis/visualisation script
np.save(
    RESULTS_PATH + 'classify_times.npy',
    classify_arr
)

np.save(
    RESULTS_PATH + 'total_times.npy',
    total_arr
)

print("    Latency data saved for visualization")


print("")
print("=" * 60)
print("DETECTION COMPLETE!")
print("=" * 60)
