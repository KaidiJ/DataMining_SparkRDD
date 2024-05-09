import sys
import csv
import itertools
import random
from pyspark import SparkContext, SparkConf


def create_hash_coeffs(num_hash_funcs, num_bins):
    """
    Creates a list of tuples with random 'a' and 'b' coefficients for hash functions.
    """
    return [(random.randint(1, num_bins - 1), random.randint(0, num_bins - 1)) for _ in range(num_hash_funcs)]


def hash_functions(user_set, hash_coeffs, num_bins):
    """
    Apply a collection of hash functions to each user in the set.
    """
    sigs = []
    for a, b in hash_coeffs:
        min_hash = min([(a * user + b) % num_bins for user in user_set])
        sigs.append(min_hash)
    return sigs


def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity between two sets.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def main(input_file, output_file):
    conf = SparkConf().setAppName("LSH Yelp Data")
    sc = SparkContext(conf=conf)

    data = sc.textFile(input_file).map(lambda line: line.split(","))

    business_user_matrix = data.map(lambda x: (x[1], x[0], 1))  # Convert stars to binary

    # Generate and broadcast the mappings to all workers
    businesses = business_user_matrix.map(lambda x: x[0]).distinct().collect()
    users = business_user_matrix.map(lambda x: x[1]).distinct().collect()

    business_to_numeric = {business: idx for idx, business in enumerate(businesses)}
    user_to_numeric = {user: idx for idx, user in enumerate(users)}

    numeric_to_business = {idx: business for business, idx in business_to_numeric.items()}  # For decoding

    business_to_numeric_bc = sc.broadcast(business_to_numeric)
    user_to_numeric_bc = sc.broadcast(user_to_numeric)

    numeric_business_user_matrix = business_user_matrix.map(
        lambda x: (business_to_numeric_bc.value[x[0]], {user_to_numeric_bc.value[x[1]]})
    ).reduceByKey(lambda a, b: a.union(b))

    num_businesses = len(business_to_numeric)
    num_hash_funcs = 100
    num_bins = 2 * num_businesses
    hash_coeffs = create_hash_coeffs(num_hash_funcs, num_bins)

    signature_matrix = numeric_business_user_matrix.map(
        lambda x: (x[0], hash_functions(x[1], hash_coeffs, num_bins))
    )

    num_bands = 20
    rows_per_band = 5

    candidate_pairs = signature_matrix.flatMap(
        lambda x: [((i, hash(tuple(x[1][i * rows_per_band:(i + 1) * rows_per_band]))), x[0]) for i in range(num_bands)]
    ).groupByKey().flatMap(
        lambda x: itertools.combinations(x[1], 2)
    ).distinct()

    business_user_sets_dict = numeric_business_user_matrix.collectAsMap()
    business_user_sets_dict_bc = sc.broadcast(business_user_sets_dict)

    similar_pairs = candidate_pairs.map(
        lambda pair: ((pair[0], pair[1]), jaccard_similarity(business_user_sets_dict_bc.value.get(pair[0], set()),
                                                             business_user_sets_dict_bc.value.get(pair[1], set())))
    ).filter(lambda x: x[1] >= 0.5)

    similar_pairs_with_original_ids = similar_pairs.map(
        lambda x: ((numeric_to_business[x[0][0]], numeric_to_business[x[0][1]]), x[1])
    )

    def sort_pair(pair):
        return tuple(sorted(pair))

    similar_pairs_sorted = similar_pairs_with_original_ids.map(
        lambda x: (sort_pair(x[0]), x[1])
    ).sortBy(lambda x: (x[0][0], x[0][1]))

    with open(output_file, 'w', newline='') as out_file:  # 确保在Windows中不会有额外的空行
        writer = csv.writer(out_file)
        writer.writerow(['business_id_1', 'business_id_2', 'similarity'])
        for pair, sim in similar_pairs_sorted.collect():
            # 确保business_id对已经是按词典序排列的
            writer.writerow([pair[0], pair[1], sim])

    sc.stop()


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
