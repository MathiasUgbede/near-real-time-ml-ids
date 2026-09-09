import socket
import json
import time
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Path to the preprocessed/selected UNSW-NB15 test data
DATA_PATH = '/home/mathias/ml-ids-project/data/'
TEST_FILE = DATA_PATH + 'test_selected.csv'

# IP address of VM2, where the IDS detector is running
VM2_IP = '192.168.56.102'

# TCP port used for communication with VM2
PORT = 9999

# Number of test records to stream
NUM_FLOWS = 1000


print("=" * 60)
print("TRAFFIC GENERATOR (VM1) - STREAMING FLOWS TO VM2")
print("=" * 60)
print("")


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("[1] Loading test data...")

# Load the prepared UNSW-NB15 test dataset
test_df = pd.read_csv(TEST_FILE)

print(
    "    Test set: "
    + str(test_df.shape[0])
    + " records available"
)


# ============================================================
# 3. SELECT 1,000 TEST RECORDS
# ============================================================

# Randomly select 1,000 test records.
# random_state=42 makes the sample reproducible.
sample = test_df.sample(
    n=NUM_FLOWS,
    random_state=42
).reset_index(drop=True)

# Separate predictive features from the actual label
X_sample = sample.drop('label', axis=1)
y_sample = sample['label']

print(
    "    Streaming "
    + str(NUM_FLOWS)
    + " flows to VM2"
)


# ============================================================
# 4. CONNECT TO VM2 IDS DETECTOR USING TCP
# ============================================================

print("")

print(
    "[2] Connecting to IDS detector at "
    + VM2_IP
    + ":"
    + str(PORT)
    + "..."
)

# AF_INET = IPv4
# SOCK_STREAM = TCP
client_socket = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)

# Connect VM1 to the IDS detector running on VM2
client_socket.connect((VM2_IP, PORT))

print("    Connected!")


# ============================================================
# 5. STREAM TEST RECORDS TO VM2
# ============================================================

print("")
print("[3] Streaming flows...")

# Start timing the sending process
send_start = time.time()


for i in range(len(X_sample)):

    # Create one record containing:
    # - feature values used by the model
    # - actual label used later for evaluation
    record = {
        'features': X_sample.iloc[i].tolist(),
        'label': int(y_sample.iloc[i])
    }

    # Convert the record into JSON and add a newline
    # so VM2 can separate individual records
    message = json.dumps(record) + "\n"

    # Send the encoded record to VM2
    client_socket.sendall(
        message.encode('utf-8')
    )

    # Display progress every 100 records
    if (i + 1) % 100 == 0:
        print(
            "    Sent "
            + str(i + 1)
            + " flows..."
        )


# ============================================================
# 6. SEND END SIGNAL
# ============================================================

# Tell VM2 that all 1,000 records have been transmitted
client_socket.sendall(
    "END\n".encode('utf-8')
)


# ============================================================
# 7. CALCULATE STREAMING TIME
# ============================================================

send_time = time.time() - send_start

print("")
print("[4] Streaming complete!")

print(
    "    Sent "
    + str(NUM_FLOWS)
    + " flows in "
    + format(send_time, '.2f')
    + " seconds"
)


# ============================================================
# 8. CLOSE CONNECTION
# ============================================================

client_socket.close()


print("")
print("=" * 60)
print("TRAFFIC GENERATION COMPLETE!")
print("=" * 60)
