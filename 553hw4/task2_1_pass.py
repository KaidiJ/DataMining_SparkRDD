from collections import defaultdict
from pyspark import SparkContext
from itertools import combinations
import sys
import queue


def compute_shortest_paths(root, adj_list):
    """计算给定根节点的最短路径和路径数量"""
    q = queue.Queue()
    q.put(root)
    visited = {root: 0}
    parents = defaultdict(list)
    num_paths = defaultdict(int)
    num_paths[root] = 1

    while not q.empty():
        node = q.get()
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                q.put(neighbor)
            if visited[neighbor] == visited[node] + 1:
                parents[neighbor].append(node)
                num_paths[neighbor] += num_paths[node]

    return parents, num_paths, visited


def compute_edge_betweenness(adj_list):
    """使用优化的回溯方法计算边介数"""
    betweenness = defaultdict(float)
    for root in adj_list:
        parents, num_paths, visited = compute_shortest_paths(root, adj_list)
        node_credit = defaultdict(float, {n: 1.0 for n in adj_list})

        sorted_nodes = sorted(visited, key=visited.get, reverse=True)
        for node in sorted_nodes:
            for parent in parents[node]:
                fraction = num_paths[parent] / num_paths[node]
                edge_credit = node_credit[node] * fraction

                # 确保边的表示是统一的，小的节点ID在前
                ordered_edge = tuple(sorted((parent, node)))

                betweenness[ordered_edge] += edge_credit

                node_credit[parent] += edge_credit

    # 介数需要除以2，因为无向图中的每条边都被计算了两次
    for edge in betweenness:
        betweenness[edge] /= 2
    return betweenness


def calculate_betweenness(edges_rdd):
    adj_list = edges_rdd.flatMap(lambda edge: [edge, (edge[1], edge[0])]) \
        .groupByKey().mapValues(set).collectAsMap()
    betweenness = compute_edge_betweenness(adj_list)
    betweenness_rdd = sc.parallelize([(edge, val) for edge, val in betweenness.items()])
    return betweenness_rdd.sortBy(lambda x: (-x[1], x[0]))


# Initialize Spark Context
sc = SparkContext.getOrCreate()
sc.setLogLevel("WARN")

# Read arguments
filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
output_file_path = sys.argv[3]
output_file_path_2 = sys.argv[4]

# Load data
rdd = sc.textFile(input_file_path)
header = rdd.first()
rdd = rdd.filter(lambda line: line != header).map(lambda line: line.split(','))

# Create user-business pairs
user_business_pairs = rdd.map(lambda x: (x[0], x[1]))

# Filter based on threshold
user_pairs = user_business_pairs.groupByKey().mapValues(set).cache()

edges = user_pairs.cartesian(user_pairs) \
    .filter(lambda x: x[0][0] < x[1][0]) \
    .filter(lambda x: len(x[0][1].intersection(x[1][1])) >= filter_threshold) \
    .map(lambda x: (x[0][0], x[1][0]))

# Calculate betweenness
betweenness = calculate_betweenness(edges)

# Sort by betweenness value in descending order, then by the first user_id lexicographically
sorted_betweenness = betweenness.sortBy(lambda x: (-x[1], x[0]))

# Output
with open(output_file_path, 'w') as file:
    for edge, betweenness_value in sorted_betweenness.collect():
        line = f"{edge}, {round(betweenness_value, 5)}\n"
        file.write(line)
