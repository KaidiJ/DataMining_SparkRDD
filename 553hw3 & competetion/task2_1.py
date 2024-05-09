from pyspark import SparkContext, SparkConf
import time, sys


def calculate_similarity(target_business, target_user, businesses_users, users_businesses, ratings_user_business,
                         avg_ratings_by_user):
    if target_user not in users_businesses:
        return 3.5

    if target_business not in businesses_users:
        return avg_ratings_by_user.get(target_user, 3.5)

    scores = {}
    for business_reviewed_by_user in users_businesses[target_user]:
        shared_users = businesses_users[target_business].intersection(businesses_users[business_reviewed_by_user])

        if len(shared_users) == 0:
            continue

        ratings_for_target = [float(ratings_user_business[target_business][user]) - avg_ratings_by_user.get(user, 3.5)
                              for user in shared_users]

    top_scores = sorted(scores.items(), key=lambda x: -x[1])[:15]

    user_avg = avg_ratings_by_user.get(target_user, 3.5)

    adjusted_weighted_sum = sum(score * ratings_for_target[i] for i, (_, score) in enumerate(top_scores))
    adjusted_weights_sum = sum(abs(score) for _, score in top_scores)

    predicted_rating = user_avg + (adjusted_weighted_sum / adjusted_weights_sum if adjusted_weights_sum else 0)
    predicted_rating = min(max(predicted_rating, 1), 5)

    return predicted_rating


def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return None


def main(train_path, validation_path, result_path):
    start_time = time.time()

    conf = SparkConf().setAppName("ItemBasedCF")
    sc = SparkContext.getOrCreate(conf=conf)

    train_header = sc.textFile(train_path).first()
    val_header = sc.textFile(validation_path).first()

    train_lines = sc.textFile(train_path).filter(lambda x: x != train_header)
    val_lines = sc.textFile(validation_path).filter(lambda x: x != val_header)

    train_rdd = train_lines.map(lambda x: x.split(',')).map(lambda x: (x[1], x[0], safe_float(x[2]))).filter(
        lambda x: x[2] is not None)

    business_to_users = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
    user_to_businesses = train_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
    business_to_user_ratings = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collectAsMap()
    avg_ratings_per_user = train_rdd.map(lambda x: (x[1], x[2])).groupByKey().mapValues(
        lambda scores: sum(scores) / len(scores)).collectAsMap()

    val_rdd = val_lines.map(lambda x: x.split(',')).map(lambda x: (x[1], x[0]))

    prediction_output = val_rdd.map(lambda
                                        x: f"{x[1]},{x[0]},{max(calculate_similarity(x[0], x[1], business_to_users, user_to_businesses, business_to_user_ratings, avg_ratings_per_user), 0)}").collect()
    prediction_text = "user_id,business_id,prediction\n" + "\n".join(prediction_output)

    with open(result_path, "w") as file:
        file.write(prediction_text)

    print(f"Process completed in {time.time() - start_time} seconds.")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: script <train_path> <validation_path> <result_path>")
        sys.exit(1)

    train_path = sys.argv[1]
    validation_path = sys.argv[2]
    result_path = sys.argv[3]
    main(train_path, validation_path, result_path)
