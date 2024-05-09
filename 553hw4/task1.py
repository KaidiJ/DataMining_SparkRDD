from pyspark.sql import SparkSession, functions as F
from graphframes import GraphFrame


def main(filter_threshold, input_file_path, community_output_file_path):
    # Initialize Spark Session
    spark = SparkSession.builder.appName("Community Detection").getOrCreate()

    # Read dataset
    df = spark.read.csv(input_file_path, header=True, inferSchema=True)

    # Set log level to WARN to reduce the amount of logging
    spark.sparkContext.setLogLevel("WARN")

    # Create pairs of users and businesses
    user_business_pairs = df.rdd.map(lambda row: (row['user_id'], row['business_id']))

    # Filter based on threshold
    user_pairs = user_business_pairs.groupByKey().mapValues(set).collectAsMap()

    # Build edges between users sharing businesses above the threshold
    edges = []
    nodes = set()
    for user1 in user_pairs:
        for user2 in user_pairs:
            if user1 < user2 and len(user_pairs[user1].intersection(user_pairs[user2])) >= filter_threshold:
                edges.append((user1, user2))
                edges.append((user2, user1))
                nodes.update([user1, user2])

    # Convert to DataFrames
    vertices_df = spark.createDataFrame([(node,) for node in nodes], ['id'])
    edges_df = spark.createDataFrame(edges, ['src', 'dst'])

    # Create GraphFrame and run Label Propagation Algorithm
    g = GraphFrame(vertices_df, edges_df)
    result = g.labelPropagation(maxIter=5)

    # Process communities
    communities = (result.groupBy('label')
                   .agg(F.collect_list('id').alias('members'))
                   .withColumn('members', F.sort_array('members'))
                   .withColumn('size', F.size('members'))
                   .orderBy('size', 'members'))

    # Save the communities to the file
    with open(community_output_file_path, 'w') as f:
        for row in communities.collect():
            members = ["'" + member + "'" for member in row['members']]
            f.write(', '.join(members) + '\n')

    spark.stop()


if __name__ == "__main__":
    import sys

    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    community_output_file_path = sys.argv[3]

    main(filter_threshold, input_file_path, community_output_file_path)
