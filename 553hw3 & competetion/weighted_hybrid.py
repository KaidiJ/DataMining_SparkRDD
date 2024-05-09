import sys
import json
from pyspark import SparkContext, SparkConf
import numpy as np
import xgboost as xgb
from statistics import pstdev, mean

def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def calculate_variance(ratings):
    if len(ratings) > 1:
        return pstdev(ratings)
    return 0

def load_data(sc, user_path, business_path):
    user_data = sc.textFile(user_path).map(json.loads).map(
        lambda x: (x['user_id'], (x['average_stars'], len(x['reviews']))))
    business_ratings = sc.textFile(business_path).flatMap(
        lambda x: [(x['business_id'], safe_float(review['stars'])) for review in x.get('reviews', [])])
    business_stats = business_ratings.aggregateByKey([], lambda a, b: a + [b], lambda a, b: a + b)
    business_stats = business_stats.mapValues(
        lambda ratings: (mean(ratings), len(ratings), calculate_variance(ratings)))
    return user_data, business_stats

def calculate_alpha(num_reviews, variance, max_reviews, min_variance, max_variance):
    normalized_reviews = num_reviews / max_reviews
    normalized_variance = (variance - min_variance) / (max_variance - min_variance) if max_variance > min_variance else 0
    return 0.5 + 0.5 * normalized_reviews * (1 - normalized_variance)

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

def load_avg_ratings(sc, user_avg_path, business_avg_path):
    # Load average ratings from JSON files into RDDs
    user_data = sc.textFile(user_avg_path).map(json.loads).map(lambda x: (x['user_id'], (x['average_stars'], x['review_count'])))
    business_data = sc.textFile(business_avg_path).map(json.loads).map(lambda x: (x['business_id'], (x['stars'], x['review_count'])))
    return user_data.cache(), business_data.cache()

def feature_engineering(rdd, user_data, business_data, include_rating=True):
    # Join user and business data with the main RDD
    if include_rating:
        # For training data which includes ratings
        train_rdd = rdd.map(lambda x: (x[1], (x[0], x[2])))  # (user_id, (business_id, rating))
        train_rdd = train_rdd.join(user_data).map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1][0], x[1][1][1])))  # (business_id, (user_id, rating, user_avg_rating, user_review_count))
        final_rdd = train_rdd.join(business_data).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1][0], x[1][1][1]))  # (business_id, user_id, rating, user_avg_rating, user_review_count, business_avg_rating, business_review_count)
    else:
        # For validation data without ratings
        val_rdd = rdd.map(lambda x: (x[1], x[0]))  # (user_id, business_id)
        final_rdd = val_rdd.join(user_data).map(lambda x: (x[1][0], (x[0], x[1][1][0], x[1][1][1])))  # (business_id, (user_id, user_avg_rating, user_review_count))
        final_rdd = final_rdd.join(business_data).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1][0], x[1][1][1]))  # (business_id, user_id, user_avg_rating, user_review_count, business_avg_rating, business_review_count)

    return final_rdd
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: script <train_path> <validation_path> <result_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    validation_path = sys.argv[2]
    result_path = sys.argv[3]

    train_path = folder_path + '/yelp_train.csv'
    business_avg_path = folder_path + '/business.json'
    user_avg_path = folder_path + '/user.json'

    conf = SparkConf().setAppName("hybrid")
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("WARN")
    # spark = SparkSession.builder.appName("hybrid").getOrCreate()

    train_header = sc.textFile(train_path).first()
    val_header = sc.textFile(validation_path).first()

    train_lines = sc.textFile(train_path).filter(lambda x: x != train_header)
    val_lines = sc.textFile(validation_path).filter(lambda x: x != val_header)

    train_rdd = train_lines.map(lambda x: x.split(',')).map(lambda x: (x[1], x[0], safe_float(x[2]))).filter(lambda x: x[2] is not None)
    val_rdd = val_lines.map(lambda x: x.split(',')).map(lambda x: (x[1], x[0]))

    user_data, business_data = load_avg_ratings(sc, user_avg_path, business_avg_path)
    train_rdd = feature_engineering(train_rdd, user_data, business_data)

    # item-based
    business_to_users = train_rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set).collectAsMap()
    user_to_businesses = train_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set).collectAsMap()
    business_to_user_ratings = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(dict).collectAsMap()
    avg_ratings_per_user = train_rdd.map(lambda x: (x[1], x[2])).groupByKey().mapValues(
        lambda scores: sum(scores) / len(scores)).collectAsMap()
    # Calculate CF predictions
    cf_predictions = val_rdd.map(lambda x: ((x[1], x[0]), max(calculate_similarity(x[0], x[1], business_to_users, user_to_businesses, business_to_user_ratings, avg_ratings_per_user), 0))).collectAsMap()

    # model-based
    # Prepare features for model-based predictions
    train_features_labels = train_rdd.map(lambda x: (x[3:], x[2])).collect()
    train_features, train_labels = zip(*train_features_labels)
    train_features = np.array(train_features, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)

    val_rdd = feature_engineering(val_rdd, user_data, business_data, include_rating=False)
    val_user_business = val_rdd.map(lambda x: (x[1], x[0])).collect()
    val_features = np.array(val_rdd.map(lambda x: x[2:]).collect(), dtype=np.float32)

    model = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.05, max_depth=15, alpha=10, n_estimators=200)
    model.fit(train_features, train_labels)
    # Train XGBoost model
    model.fit(train_features, train_labels)
    xgb_predictions = model.predict(val_features)

    # Calculate business statistics for dynamic alpha
    business_stats = train_rdd.map(lambda x: (x[0], x[2])).groupByKey().mapValues(
        lambda ratings: (len(ratings), np.var(list(ratings), ddof=1 if len(ratings) > 1 else 0)))
    business_stats_dict = dict(business_stats.collect())

    # Combine CF and XGBoost predictions dynamically adjusting alpha
    final_predictions = []
    for (user_id, business_id), xgb_pred in zip(val_user_business, xgb_predictions):
        num_reviews, variance = business_stats_dict.get(business_id, (0, 0))
        if num_reviews > 50:
            alpha = 0.95
        elif num_reviews > 10:
            alpha = 0.75
        else:
            alpha = 0.5  # Increase the weight of CF for businesses with fewer reviews
        cf_pred = cf_predictions.get((user_id, business_id), 3.5)  # Default to a neutral rating if not found
        final_pred = alpha * xgb_pred + (1 - alpha) * cf_pred
        final_predictions.append(f"{user_id},{business_id},{final_pred}")

    # Write combined predictions to file
    with open(result_path, "w") as file:
        file.write("user_id,business_id,prediction\n")
        file.writelines("\n".join(final_predictions))

    # Clean up Spark context
    sc.stop()
