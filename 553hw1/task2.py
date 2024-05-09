from pyspark import SparkContext, SparkConf
import json
import sys
import time

def get_partition_info(rdd):
    """
    Helper function to retrieve the number of partitions and items per partition.
    """
    return (rdd.getNumPartitions(), rdd.glom().map(len).collect())

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: spark-submit --executor-memory 4G --driver-memory 4G task2.py <review_filepath> <output_filepath> <n_partition>",
            file=sys.stderr)
        exit(-1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_partition = int(sys.argv[3])  # Number of partitions specified by the user

    conf = SparkConf().setAppName("Data Exploration with Custom Partitioner")
    sc = SparkContext(conf=conf)

    # Load the data
    reviews = sc.textFile(input_path).map(json.loads)

    # Default partitioning
    start_time = time.time()
    default_part = reviews.map(lambda review: (review['business_id'], 1)).reduceByKey(lambda a, b: a + b)
    default_count = default_part.sortBy(lambda x: (-x[1], x[0])).take(10)
    default_n_partition, default_n_items = get_partition_info(default_part)
    default_exe_time = time.time() - start_time

    # Custom partitioning
    start_time = time.time()
    custom_part = reviews.map(lambda review: (review['business_id'], 1)) \
        .reduceByKey(lambda a, b: a + b, numPartitions=n_partition)
    custom_count = custom_part.sortBy(lambda x: (-x[1], x[0])).take(10)
    custom_n_partition, custom_n_items = get_partition_info(custom_part)
    custom_exe_time = time.time() - start_time

    # Prepare the output
    output = {
        "default": {
            "n_partition": default_n_partition,
            "n_items": default_n_items,
            "exe_time": default_exe_time
        },
        "customized": {
            "n_partition": custom_n_partition,
            "n_items": custom_n_items,
            "exe_time": custom_exe_time
        }
    }

    with open(output_path, 'w') as outfile:
        json.dump(output, outfile)

    sc.stop()