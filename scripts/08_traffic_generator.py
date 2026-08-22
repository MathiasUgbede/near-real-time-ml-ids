import socket
import json
import time
import pandas as pd

DATA_PATH = '/home/mathias/ml-ids-project/data/'
TEST_FILE = DATA_PATH + 'test_selected.csv'
VM2_IP = '192.168.56.102'
PORT = 9999
NUM_FLOWS = 1000

print("=" * 60)
print("TRAFFIC GENERATOR (VM1) - STREAMING FLOWS TO VM2")
print("=" * 60)
print("")

print("[1] Loading test data...")
test_df = pd.read_csv(TEST_FILE)
print("    Test set: " + str(test_df.shape[0]) + " records available")

sample = test_df.sample(n=NUM_FLOWS, random_state=42).reset_index(drop=True)
X_sample = sample.drop('label', axis=1)
y_sample = sample['label']
print("    Streaming " + str(NUM_FLOWS) + " flows to VM2")

print("")
print("[2] Connecting to IDS detector at " + VM2_IP + ":" + str(PORT) + "...")
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((VM2_IP, PORT))
print("    Connected!")

print("")
print("[3] Streaming flows...")
send_start = time.time()

for i in range(len(X_sample)):
    record = {
        'features': X_sample.iloc[i].tolist(),
        'label': int(y_sample.iloc[i])
    }
    message = json.dumps(record) + "\n"
    client_socket.sendall(message.encode('utf-8'))
    if (i + 1) % 100 == 0:
        print("    Sent " + str(i + 1) + " flows...")

client_socket.sendall("END\n".encode('utf-8'))
send_time = time.time() - send_start

print("")
print("[4] Streaming complete!")
print("    Sent " + str(NUM_FLOWS) + " flows in " + format(send_time, '.2f') + " seconds")

client_socket.close()

print("")
print("=" * 60)
print("TRAFFIC GENERATION COMPLETE!")
print("=" * 60)
