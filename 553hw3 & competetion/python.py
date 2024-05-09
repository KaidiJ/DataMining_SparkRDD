import sys
from pyspark import SparkContext, SparkConf
import numpy as np
import xgboost as xgb
import json
from math import sqrt
from pyspark.sql import SparkSession

def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def load_avg_ratings(sc, user_avg_path, business_avg_path):
    user_data = sc.textFile(user_avg_path).map(json.loads).map(lambda user: (user['user_id'], (
            user['average_stars'], user['review_count'], user['useful']**5, user['funny']**5, user['cool']**5, user['fans']**5,
            sum(x ** 1.2 for x in [
                user['compliment_hot'], user['compliment_more'], user['compliment_profile'], user['compliment_cute'],
                user['compliment_list'], user['compliment_note'], user['compliment_plain'], user['compliment_cool'],
                user['compliment_funny'], user['compliment_writer'], user['compliment_photos']
            ]),
            1 if user['elite'] else 0
        )))
    business_data = sc.textFile(business_avg_path).map(json.loads).map(lambda business: (business['business_id'], (
        business['stars'], business['review_count'],
        int(business['attributes']['GoodForKids'] == 'True') if business['attributes'] and 'GoodForKids' in business['attributes'] else 0,
        int(business['attributes']['DriveThru'] == 'True') if business['attributes'] and 'DriveThru' in business['attributes'] else 0,
        int(business['attributes']['RestaurantsPriceRange2']) if business['attributes'] and 'RestaurantsPriceRange2' in business['attributes'] else 0,
        int(business['attributes']['RestaurantsReservations'] == 'True') if business['attributes'] and 'RestaurantsReservations' in business['attributes'] else 0,
        int(business['attributes']['RestaurantsTableService'] == 'True') if business['attributes'] and 'RestaurantsTableService' in business['attributes'] else 0
    )))
    return user_data.cache(), business_data.cache()


def feature_engineering(rdd, user_data, business_data, include_rating=True):
    if include_rating:
        train_rdd = rdd.map(lambda x: (x[1], (x[0], x[2])))  # (user_id, (business_id, rating))
        train_rdd = train_rdd.leftOuterJoin(user_data).map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1] if x[1][1] else (0.0, 0, 0, 0, 0, 0))))  # 使用默认值填充用户数据
        final_rdd = train_rdd.leftOuterJoin(business_data).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2][0], x[1][0][2][1], x[1][0][2][2], x[1][0][2][3], x[1][0][2][4], x[1][0][2][5], x[1][0][2][6], x[1][0][2][7], x[1][1][0] if x[1][1] else 0.0, x[1][1][1] if x[1][1] else 0, x[1][1][2] if x[1][1] else 0, x[1][1][3] if x[1][1] else 0, x[1][1][4] if x[1][1] else 0, x[1][1][5] if x[1][1] else 0, x[1][1][6] if x[1][1] else 0))  # 使用默认值填充商家数据
    else:
        val_rdd = rdd.map(lambda x: (x[1], x[0]))  # (user_id, business_id)
        final_rdd = val_rdd.leftOuterJoin(user_data).map(lambda x: (x[1][0], (x[0], x[1][1] if x[1][1] else (0.0, 0, 0, 0, 0, 0))))  # 使用默认值填充用户数据
        final_rdd = final_rdd.leftOuterJoin(business_data).map(lambda x: (x[0], x[1][0][0], x[1][0][1][0], x[1][0][1][1], x[1][0][1][2], x[1][0][1][3], x[1][0][1][4], x[1][0][1][5], x[1][0][1][6], x[1][0][1][7], x[1][1][0] if x[1][1] else 0.0, x[1][1][1] if x[1][1] else 0, x[1][1][2] if x[1][1] else 0, x[1][1][3] if x[1][1] else 0, x[1][1][4] if x[1][1] else 0, x[1][1][5] if x[1][1] else 0, x[1][1][6] if x[1][1] else 0))  # 使用默认值填充商家数据
    return final_rdd

if __name__ == '__main__':
    folder_path = sys.argv[1]
    validation_path = sys.argv[2]
    result_path = sys.argv[3]

    train_path = folder_path + '/yelp_train.csv'
    business_avg_path = folder_path + '/business.json'
    user_avg_path = folder_path + '/user.json'

    # Initialize Spark Session
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

    # Convert RDD to NumPy arrays
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

    # 定义新的参数
    param = {
        'lambda': 9.92724463758443,
        'alpha': 0.2765119705933928,
        'colsample_bytree': 0.5,
        'subsample': 0.8,
        'learning_rate': 0.02,
        'max_depth': 17,
        'random_state': 2020,
        'min_child_weight': 101,
        'n_estimators': 320,
    }

    # 使用参数初始化XGBRegressor
    xgb_model = xgb.XGBRegressor(**param)

    # 使用训练数据集进行模型训练
    xgb_model.fit(train_features, train_labels)

    # 使用验证集进行预测
    predictions = xgb_model.predict(val_features)

    # Load validation ratings
    spark = SparkSession.builder.appName("ModelBasedCF").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    val_data = spark.read.csv(validation_path, header=True, inferSchema=True).rdd \
        .map(lambda row: ((row['user_id'], row['business_id']), row['stars']))

    # Collect predictions as RDD
    predictions_rdd = spark.sparkContext.parallelize(
        [(user_business, prediction) for user_business, prediction in zip(val_user_business, predictions)]
    )

    # Join predictions with true ratings
    predictions_and_labels = predictions_rdd.join(val_data)

    # Compute RMSE
    mse = predictions_and_labels.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean()
    rmse = sqrt(mse)

    print(f"RMSE: {rmse}")

    # Write predictions to file
    with open(result_path, "w") as file:
        file.write("user_id,business_id,prediction\n")
        for (user_id, business_id), prediction in zip(val_user_business, predictions):
            file.write(f"{user_id},{business_id},{prediction}\n")

# RMSE: 0.980279487608425     320
#0.9809
