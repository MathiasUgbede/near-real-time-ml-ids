import socket
import json
import time
import warnings
import numpy as np
import joblib

warnings.filterwarnings("ignore")

MODEL_PATH = '/home/mathias/ml-ids-project/models/random_forest_model.pkl'
RESULTS_PATH = '/home/mathias/ml-ids-project/results/'
HOST = '0.0.0.0'
PORT = 9999
WARMUP = 10  # discard first N flows (model warm-up) from statistics

print("=" * 60)
print("IDS DETECTOR (VM2) - REAL-TIME CLASSIFICATION")
print("=" * 60)
print("")

print("[1] Loading Random Forest model...")
model = joblib.load(MODEL_PATH)
print("    Model loaded successfully")

print("")
print("[2] Starting detector on port " + str(PORT) + "...")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("    Waiting for connection from VM1...")

conn, addr = server_socket.accept()
print("    Connected to traffic generator at " + str(addr[0]))

print("")
print("[3] Receiving and classifying flows...")
classify_times = []   # pure model prediction time
total_times = []      # arrival-to-decision time
predictions = []
actuals = []
flow_count = 0
buffer = ""
start_session = time.time()

try:
    while True:
        data = conn.recv(4096).decode('utf-8')
        if not data:
            break
        buffer += data
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if line.strip() == "END":
                raise StopIteration
            if not line.strip():
                continue

            arrival_time = time.time()
            record = json.loads(line)
            features = np.array(record['features']).reshape(1, -1)
            actual_label = record['label']

            classify_start = time.time()
            prediction = model.predict(features)[0]
            classify_end = time.time()

            classify_ms = (classify_end - classify_start) * 1000
            total_ms = (classify_end - arrival_time) * 1000

            classify_times.append(classify_ms)
            total_times.append(total_ms)
            predictions.append(int(prediction))
            actuals.append(int(actual_label))

            flow_count += 1
            if flow_count % 100 == 0:
                print("    Processed " + str(flow_count) + " flows...")
except StopIteration:
    print("    Received END signal from VM1")

session_time = time.time() - start_session
conn.close()
server_socket.close()

# Discard warm-up flows from statistics
classify_arr = np.array(classify_times[WARMUP:])
total_arr = np.array(total_times[WARMUP:])
kept = len(classify_arr)

print("")
print("[4] Classification Latency (pure model prediction):")
print("    Flows in statistics: " + str(kept) + " (first " + str(WARMUP) + " discarded as warm-up)")
print("    Average: " + format(np.mean(classify_arr), '.4f') + " ms")
print("    Median:  " + format(np.median(classify_arr), '.4f') + " ms")
print("    Min:     " + format(np.min(classify_arr), '.4f') + " ms")
print("    Max:     " + format(np.max(classify_arr), '.4f') + " ms")
print("    Std dev: " + format(np.std(classify_arr), '.4f') + " ms")

print("")
print("[5] End-to-End Latency (arrival to decision):")
print("    Average: " + format(np.mean(total_arr), '.4f') + " ms")
print("    Median:  " + format(np.median(total_arr), '.4f') + " ms")
print("    Min:     " + format(np.min(total_arr), '.4f') + " ms")
print("    Max:     " + format(np.max(total_arr), '.4f') + " ms")
print("    Std dev: " + format(np.std(total_arr), '.4f') + " ms")

throughput = flow_count / session_time if session_time > 0 else 0
print("")
print("    Total session time: " + format(session_time, '.2f') + " seconds")
print("    Throughput: " + format(throughput, '.1f') + " flows per second")

correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
accuracy = correct / flow_count if flow_count > 0 else 0
print("")
print("[6] Detection accuracy on streamed flows: " + format(accuracy*100, '.2f') + "%")

print("")
print("[7] Saving results...")
results_file = RESULTS_PATH + 'realtime_detection_results.txt'
with open(results_file, 'w') as f:
    f.write("TWO-VM REAL-TIME DETECTION RESULTS (Random Forest)\n")
    f.write("=" * 50 + "\n")
    f.write("Flows processed (total): " + str(flow_count) + "\n")
    f.write("Flows in statistics: " + str(kept) + " (warm-up " + str(WARMUP) + " discarded)\n")
    f.write("\n-- Classification latency (pure model) --\n")
    f.write("Average: " + format(np.mean(classify_arr), '.4f') + " ms\n")
    f.write("Median:  " + format(np.median(classify_arr), '.4f') + " ms\n")
    f.write("Min:     " + format(np.min(classify_arr), '.4f') + " ms\n")
    f.write("Max:     " + format(np.max(classify_arr), '.4f') + " ms\n")
    f.write("Std dev: " + format(np.std(classify_arr), '.4f') + " ms\n")
    f.write("\n-- Detector-Side Processing Latency (arrival to decision) --\n")
    f.write("Average: " + format(np.mean(total_arr), '.4f') + " ms\n")
    f.write("Median:  " + format(np.median(total_arr), '.4f') + " ms\n")
    f.write("Min:     " + format(np.min(total_arr), '.4f') + " ms\n")
    f.write("Max:     " + format(np.max(total_arr), '.4f') + " ms\n")
    f.write("Std dev: " + format(np.std(total_arr), '.4f') + " ms\n")
    f.write("\nThroughput: " + format(throughput, '.1f') + " flows per second\n")
    f.write("Detection accuracy: " + format(accuracy*100, '.2f') + "%\n")
print("    Saved: " + results_file)

np.save(RESULTS_PATH + 'classify_times.npy', classify_arr)
np.save(RESULTS_PATH + 'total_times.npy', total_arr)
print("    Latency data saved for visualization")

print("")
print("=" * 60)
print("DETECTION COMPLETE!")
print("=" * 60)
