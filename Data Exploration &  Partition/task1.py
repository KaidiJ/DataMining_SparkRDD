from pyspark import SparkContext
import json
import sys

sc = SparkContext(appName="Data Exploration")

input_path = sys.argv[1]
output_path = sys.argv[2]

reviews = sc.textFile(input_path).map(json.loads)

# A
total_reviews = reviews.count()

# B
reviews_2018 = reviews.filter(lambda review: review['date'].startswith("2018")).count()

# C
distinct_users = reviews.map(lambda review: review['user_id']).distinct().count()

# D
top_10_users = reviews.map(lambda review: (review['user_id'], 1)) \
                      .reduceByKey(lambda a, b: a + b) \
                      .sortBy(lambda x: (-x[1], x[0])) \
                      .take(10)

# E
distinct_businesses = reviews.map(lambda review: review['business_id']).distinct().count()

# F
top_10_businesses = reviews.map(lambda review: (review['business_id'], 1)) \
                           .reduceByKey(lambda a, b: a + b) \
                           .sortBy(lambda x: (-x[1], x[0])) \
                           .take(10)

output = {
    "n_review": total_reviews,
    "n_review_2018": reviews_2018,
    "n_user": distinct_users,
    "top10_user": [[user, count] for user, count in top_10_users],
    "n_business": distinct_businesses,
    "top10_business": [[business, count] for business, count in top_10_businesses]
}

with open(output_path, 'w') as outfile:
    json.dump(output, outfile)

sc.stop()
