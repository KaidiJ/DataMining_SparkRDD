import sys
import time
from pyspark import SparkContext
from itertools import chain
from collections import defaultdict

def generate_candidates(frequent_itemsets, k):
    """Efficiently generate candidate itemsets of size k based on previous frequent itemsets."""
    frequent_items = set(chain.from_iterable(frequent_itemsets))
    candidates = set()
    for itemset in frequent_itemsets:
        for item in frequent_items:
            if item not in itemset:
                candidate = tuple(sorted(itemset + (item,)))
                if len(candidate) == k:
                    candidates.add(candidate)
    return candidates

def get_frequent_itemsets(transactions, candidates, support_threshold):
    """Return itemsets that meet the support threshold."""
    itemset_counts = defaultdict(int)
    transactions = [set(t) for t in transactions]  # Convert transactions to sets for faster operations
    for candidate in candidates:
        candidate_set = set(candidate)
        for transaction in transactions:
            if candidate_set.issubset(transaction):
                itemset_counts[candidate] += 1

    return {itemset for itemset, count in itemset_counts.items() if count >= support_threshold}

def write_output(output_file, candidates, frequent_itemsets):
    with open(output_file, 'w') as f:
        f.write("Candidates:\n")
        for k, itemsets in sorted(candidates.items()):
            # Sort the itemsets within each pass and remove single item trailing comma
            sorted_itemsets = sorted(['(' + ', '.join(map(lambda x: f"'{x}'", item)) + ')' for item in itemsets])
            f.write(', '.join(sorted_itemsets) + "\n\n")

        f.write("Frequent Itemsets:\n")
        for k, itemsets in sorted(frequent_itemsets.items()):
            # Sort the itemsets within each pass and remove single item trailing comma
            sorted_itemsets = sorted(['(' + ', '.join(map(lambda x: f"'{x}'", item)) + ')' for item in itemsets])
            f.write(', '.join(sorted_itemsets) + "\n\n")

def main(argv):
    start_time = time.time()

    filter_threshold = int(argv[1])
    support = int(argv[2])
    input_file_path = argv[3]
    output_file_path = argv[4]

    sc = SparkContext("local", "SONAlgorithm")
    data_rdd = sc.textFile(input_file_path)

    header = data_rdd.first()
    data_rdd = data_rdd.filter(lambda line: line != header)

    def parse_line(line):
        fields = line.split(',')
        transaction_date = fields[0].strip('"')
        customer_id = fields[1].strip('"').lstrip('0')
        product_id = fields[5].strip('"').lstrip('0')
        date_customer_id = f"{transaction_date}-{customer_id}"
        return (date_customer_id, product_id)
    parsed_rdd = data_rdd.map(parse_line)
    for basket in parsed_rdd.take(1):
        print(basket)

    def to_basket(items):
        return list(set(items))

    baskets_rdd = parsed_rdd.groupByKey().mapValues(to_basket) #('DATE-CUSTOMER_ID', [PRODUCT_IDs])
    filtered_baskets_rdd = baskets_rdd.filter(lambda basket: len(basket[1]) > filter_threshold)
    values_rdd = filtered_baskets_rdd.map(lambda x: x[1])
    final_list = values_rdd.collect()

    # SON
    candidates = {}
    frequent_itemsets = {}
    k = 1
    current_frequent_itemsets = {frozenset([t]) for transaction in final_list for t in transaction}

    while current_frequent_itemsets:
        candidates[k] = {tuple(itemset) for itemset in current_frequent_itemsets}
        current_frequent_itemsets = get_frequent_itemsets(final_list, candidates[k], support)
        frequent_itemsets[k] = {tuple(itemset) for itemset in current_frequent_itemsets}
        k += 1
        current_frequent_itemsets = generate_candidates(current_frequent_itemsets, k)

    write_output(output_file_path, candidates, frequent_itemsets)
    duration = time.time() - start_time
    print(f"Duration: {duration}")

if __name__ == "__main__":
    main(sys.argv)