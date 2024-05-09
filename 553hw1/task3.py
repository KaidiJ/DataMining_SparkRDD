from pyspark import SparkContext
import sys
import time
import json

# Initialize SparkContext
sc = SparkContext(appName="Data Exploration")

review_filepath = sys.argv[1]
business_filepath = sys.argv[2]
output_filepath_question_a = sys.argv[3]
output_filepath_question_b = sys.argv[4]

# Loading and transforming data to RDD
start_time_loading = time.time()

# Load reviews data, select required fields and create a pair RDD
reviews_RDD = sc.textFile(review_filepath).map(json.loads).map(lambda x: (x['business_id'], x['stars']))

# Load businesses data, select required fields and create a pair RDD
businesses_RDD = sc.textFile(business_filepath).map(json.loads).map(lambda x: (x['business_id'], x['city']))

# Join reviews and businesses RDDs on business_id
joined_RDD = reviews_RDD.join(businesses_RDD).map(lambda x: (x[1][1], x[1][0]))  # Results in (city, stars)

loading_time = time.time() - start_time_loading

# M1: Calculate average stars, collect, sort in Python, and print top 10
start_time_m1 = time.time()
average_stars_by_city_m1 = joined_RDD \
    .map(lambda x: (x[0], (x[1], 1))) \
    .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \
    .mapValues(lambda x: x[0]/x[1]) \
    .collect()

# Sort the results in Python and take top 10
sorted_average_stars_m1 = sorted(average_stars_by_city_m1, key=lambda x: (-x[1], x[0]))[:10]

m1_time = time.time() - start_time_m1 + loading_time

# M2: Using Spark's RDD to calculate average stars, sort, and collect top 10
start_time_m2 = time.time()
average_stars_by_city_m2 = joined_RDD \
    .map(lambda x: (x[0], (x[1], 1))) \
    .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \
    .mapValues(lambda x: x[0]/x[1]) \
    .sortBy(lambda x: (-x[1], x[0])) \
    .take(10)

m2_time = time.time() - start_time_m2 + loading_time

# Sort the entire dataset and save to file for Question A
sorted_average_stars = joined_RDD \
    .map(lambda x: (x[0], (x[1], 1))) \
    .reduceByKey(lambda x, y: (x[0]+y[0], x[1]+y[1])) \
    .mapValues(lambda x: x[0]/x[1]) \
    .sortBy(lambda x: (-x[1], x[0])) \
    .collect()

with open(output_filepath_question_a, 'w') as file_a:
    file_a.write("city,stars\n")
    for row in sorted_average_stars:
        file_a.write(f"{row[0]},{row[1]}\n")

# Write the execution times and top 10 results to a JSON file for Question B
execution_times = {
    "m1": m1_time,
    "m2": m2_time,
    "reason": "Comparing performance between sorting in Python (M1) and using Spark's RDD operations (M2). Spark's in-memory cluster computing is usually faster for distributed data processing compared to collecting data and processing it in Python."
}

with open(output_filepath_question_b, 'w') as f:
    json.dump(execution_times, f)

sc.stop()
