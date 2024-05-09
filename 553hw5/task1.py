from blackbox import BlackBox
import binascii, csv, random, sys

# Function to convert user_id string to integer
filter_bit_array = [0] * 69997
previous_users_set = set()
random.seed(553)  # Ensure reproducibility
a = random.sample(range(1, 69997), 10)  # Hash function parameters
b = random.sample(range(1, 69997), 10)  # Hash function parameters
p = int(1e9 + 7)  # Large prime


def string_to_int(user):
    """Convert user_id string to integer."""
    return int(binascii.hexlify(user.encode('utf8')), 16)


def myhashs(user):
    """Generate hash values for user."""
    x = string_to_int(user)
    return [((a[i] * x + b[i]) % p) % 69997 for i in range(len(a))]  # Use len(a) as the range


def check_and_update_bloom_filter(user_id):
    """Check and update the Bloom filter."""
    global filter_bit_array
    hash_results = myhashs(user_id)
    user_seen = all(filter_bit_array[hash_val] for hash_val in hash_results)
    if not user_seen:
        for hash_val in hash_results:
            filter_bit_array[hash_val] = 1
    return user_seen


def calculate_fpr(new_users, previous_users):
    """Calculate the False Positive Rate (FPR)."""
    checked_users = new_users - previous_users  # Only consider users not already in the previous_users_set
    false_positives = sum(1 for user in checked_users if check_and_update_bloom_filter(user))
    return false_positives / len(checked_users) if checked_users else 0


def bloom_filter_simulation(num_of_asks, stream_size, output_file):
    """Simulate the Bloom filter."""
    global previous_users_set
    bx = BlackBox()
    results = []

    for i in range(num_of_asks):
        stream_users = bx.ask(input_path, stream_size)
        new_users = set(stream_users)

        # Calculate false positives
        false_positives = sum(1 for user_id in new_users if user_id in previous_users_set)

        # Update the Bloom filter
        for user_id in new_users:
            check_and_update_bloom_filter(user_id)

        # Update the set of previous users
        previous_users_set.update(new_users)

        # Calculate FPR
        fpr = false_positives / stream_size  # Assuming that stream_size users are checked in each ask
        results.append((i, fpr))

    # Write the results to a CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Time', 'FPR'])
        for time, fpr in results:
            csvwriter.writerow([time, fpr])


# Entry point for the script
if __name__ == '__main__':
    input_path = sys.argv[1]  # Path to the input file
    stream_size = int(sys.argv[2])  # Size of each stream of users to check
    num_of_asks = int(sys.argv[3])  # Number of times to ask for stream users
    output_file = sys.argv[4]  # Output CSV file to write results

    bloom_filter_simulation(num_of_asks, stream_size, output_file)
