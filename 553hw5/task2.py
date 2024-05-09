import random
import csv
import sys

def string_to_int(user_id):
    """Converts a string user_id to an integer."""
    return int(binascii.hexlify(user_id.encode('utf8')), 16)

def myhashs(user):
    """Generates hash values for the user."""
    p = int(1e9 + 7)
    result = []
    x = string_to_int(user)
    for i in range(len(a)):
        result.append(((a[i] * x + b[i]) % p) % 997)
    return result

def trailing_zeros(n):
    """Counts the number of trailing zeros in the binary representation of n."""
    s = bin(n)
    return len(s) - len(s.rstrip('0'))

def flajolet_martin(stream_users):
    """Implements the Flajolet-Martin algorithm for estimating the count of unique users."""
    max_zeros = [0] * len(a)
    for user_id in stream_users:
        hash_values = myhashs(user_id)
        for i, hash_value in enumerate(hash_values):
            max_zeros[i] = max(max_zeros[i], trailing_zeros(hash_value))
    estimate = 2 ** (sum(max_zeros) / len(max_zeros))
    return estimate

# Function to perform reservoir sampling
def reservoir_sampling(stream_users, reservoir, sequence_num):
    for user in stream_users:
        sequence_num += 1  # Increment the global sequence number
        if sequence_num <= 100:
            reservoir.append(user)
        else:
            # Determine whether to keep the nth user
            if random.random() < 100 / sequence_num:
                # Randomly pick an index to replace
                replace_index = random.randint(0, 99)
                reservoir[replace_index] = user
    return reservoir, sequence_num

# Initialize the random seed
random.seed(553)

# Initialize global variables
reservoir = []
sequence_num = 0

if __name__ == '__main__':
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    # Open the input file to read users and output file to write the reservoir state
    with open(input_path, 'r') as infile, open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Seqnum', '0_id', '20_id', '40_id', '60_id', '80_id'])

        for _ in range(num_of_asks):
            # Simulate streaming by reading from the file
            stream_users = [infile.readline().strip() for _ in range(stream_size)]
            reservoir, sequence_num = reservoir_sampling(stream_users, reservoir, sequence_num)

            # Output to CSV every 100 users
            if sequence_num % 100 == 0:
                # Extract required user ids from the reservoir based on their positions
                output_row = [sequence_num] + [reservoir[i] if i < len(reservoir) else '' for i in [0, 20, 40, 60, 80]]
                csvwriter.writerow(output_row)
