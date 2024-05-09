import sys
from pyspark import SparkContext, SparkConf
import numpy as np
import xgboost as xgb
import json

def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def parse_business_attributes(attrs):
    # 检查 attrs 是否为 None 或不是字典类型
    if attrs is None or not isinstance(attrs, dict):
        return (0, 0)  # 返回默认值，表示没有自行车停车和不接受信用卡

    bike_parking = attrs.get("BikeParking", "False") == "True"
    accepts_credit = attrs.get("BusinessAcceptsCreditCards", "False") == "True"

    return (int(bike_parking), int(accepts_credit))


def load_avg_ratings(sc, user_avg_path, business_avg_path):
    # Load average ratings from JSON files into RDDs
    user_data = sc.textFile(user_avg_path).map(json.loads).map(lambda x: (x['user_id'], (x['average_stars'], x['review_count'])))
    business_data = sc.textFile(business_avg_path).map(json.loads).map(lambda x: (x['business_id'], (x['stars'], x['review_count'], parse_business_attributes(x.get('attributes', '{}')))))
    return user_data.cache(), business_data.cache()

def feature_engineering(rdd, user_data, business_data, include_rating=True):
    # Join user and business data with the main RDD
    if include_rating:
        train_rdd = rdd.map(lambda x: (x[1], (x[0], x[2])))  # (user_id, (business_id, rating))
        train_rdd = train_rdd.join(user_data).map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1][0], x[1][1][1])))  # (business_id, (user_id, rating, user_avg_rating, user_review_count))
        final_rdd = train_rdd.join(business_data).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1][0], x[1][1][1], x[1][1][2][0], x[1][1][2][1]))  # (business_id, user_id, rating, user_avg_rating, user_review_count, business_avg_rating, business_review_count, bike_parking, accepts_credit)
    else:
        val_rdd = rdd.map(lambda x: (x[1], x[0]))  # (user_id, business_id)
        final_rdd = val_rdd.join(user_data).map(lambda x: (x[1][0], (x[0], x[1][1][0], x[1][1][1])))  # (business_id, (user_id, user_avg_rating, user_review_count))
        final_rdd = final_rdd.join(business_data).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1][0], x[1][1][1], x[1][1][2][0], x[1][1][2][1]))  # (business_id, user_id, user_avg_rating, user_review_count, business_avg_rating, business_review_count, bike_parking, accepts_credit)

    return final_rdd

# Remaining part of the main function and model training stays the same
if __name__ == '__main__':
    folder_path = sys.argv[1]
    validation_path = sys.argv[2]
    result_path = sys.argv[3]

    train_path = folder_path + '/yelp_train.csv'
    business_avg_path = folder_path + '/business.json'
    user_avg_path = folder_path + '/user.json'

    conf = SparkConf().setAppName("ModelBasedCF")
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("WARN")

    # Load and preprocess training data
    train_header = sc.textFile(train_path).first()
    train_rdd = sc.textFile(train_path) \
        .filter(lambda x: x != train_header) \
        .map(lambda x: x.split(',')) \
        .map(lambda x: (x[1], x[0], safe_float(x[2]))) \
        .filter(lambda x: x[2] is not None)

    user_data, business_data = load_avg_ratings(sc, user_avg_path, business_avg_path)
    train_rdd = feature_engineering(train_rdd, user_data, business_data)

    # Convert RDD to NumPy arrays for model training
    train_features_labels = train_rdd.map(lambda x: (x[3:], x[2])).collect()
    train_features, train_labels = zip(*train_features_labels)
    train_features = np.array(train_features, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.float32)

    # Load and preprocess validation data, same logic as training data but without ratings
    val_header = sc.textFile(validation_path).first()
    val_rdd = sc.textFile(validation_path) \
        .filter(lambda x: x != val_header) \
        .map(lambda x: x.split(',')) \
        .map(lambda x: (x[1], x[0]))  # Only business_id and user_id

    val_rdd = feature_engineering(val_rdd, user_data, business_data, include_rating=False)
    val_user_business = val_rdd.map(lambda x: (x[1], x[0])).collect()
    val_features = np.array(val_rdd.map(lambda x: x[2:]).collect(), dtype=np.float32)

    # Train XGBoost model
    model = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.05, max_depth=15, alpha=10, n_estimators=200)
    model.fit(train_features, train_labels)

    # Make predictions
    predictions = model.predict(val_features)

    # Write predictions to file
    with open(result_path, "w") as file:
        file.write("user_id,business_id,prediction\n")
        for (user_id, business_id), prediction in zip(val_user_business, predictions):
            file.write(f"{user_id},{business_id},{prediction}\n")

# 0.9901020682433741,good