import sys
import csv
import time
from collections import defaultdict
from itertools import chain, combinations

def generate_candidates(frequent_itemsets, k):
    return set(combinations(set(chain.from_iterable(frequent_itemsets)), k))

def get_frequent_itemsets(transactions, candidates, support_threshold):
    itemset_counts = defaultdict(int)
    for transaction in transactions:
        for candidate in candidates:
            if all(item in transaction for item in candidate):
                itemset_counts[candidate] += 1

    return {itemset for itemset, count in itemset_counts.items() if count >= support_threshold}

def write_output(output_file, candidates, frequent_itemsets):
    with open(output_file, 'w') as f:
        f.write("Candidates:\n")
        for k, itemsets in sorted(candidates.items()):
            sorted_itemsets = sorted(
                ['(' + ', '.join(sorted(map(lambda x: f"'{x}'", item), key=str)) + ')' for item in itemsets], key=str)

            f.write(', '.join(sorted_itemsets) + "\n\n")

        f.write("Frequent Itemsets:\n")
        for k, itemsets in sorted(frequent_itemsets.items()):
            sorted_itemsets = sorted(
                ['(' + ', '.join(sorted(map(lambda x: f"'{x}'", item), key=str)) + ')' for item in itemsets], key=str)

            f.write(', '.join(sorted_itemsets) + "\n\n")

def main(case_number, support_threshold, input_file, output_file):
    start_time = time.time()
    transactions = defaultdict(set)
    with open(input_file, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            if case_number == 1:
                transactions[row[0]].add(row[1])
            else:
                transactions[row[1]].add(row[0])
    transactions = list(transactions.values())

    # Initialize
    candidates = {}
    frequent_itemsets = {}
    k = 1
    current_frequent_itemsets = {frozenset([t]) for transaction in transactions for t in transaction}

    while current_frequent_itemsets:
        candidates[k] = {tuple(itemset) for itemset in current_frequent_itemsets}
        current_frequent_itemsets = get_frequent_itemsets(transactions, candidates[k], support_threshold)
        frequent_itemsets[k] = {tuple(itemset) for itemset in current_frequent_itemsets}
        k += 1
        current_frequent_itemsets = generate_candidates(current_frequent_itemsets, k)

    write_output(output_file, candidates, frequent_itemsets)
    duration = time.time() - start_time
    print(f"Duration: {duration}")

if __name__ == "__main__":
    case_number = int(sys.argv[1])
    support_threshold = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]
    main(case_number, support_threshold, input_file_path, output_file_path)