from collections import defaultdict
from pyspark import SparkContext
import sys
import queue

def compute_shortest_paths(root, adj_list):
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
    betweenness = defaultdict(float)
    for root in adj_list:
        if root in adj_list:
            parents, num_paths, visited = compute_shortest_paths(root, adj_list)
            node_credit = defaultdict(float, {n: 1.0 for n in adj_list})
            sorted_nodes = sorted(visited, key=visited.get, reverse=True)
            for node in sorted_nodes:
                for parent in parents[node]:
                    fraction = num_paths[parent] / num_paths[node]
                    edge_credit = node_credit[node] * fraction
                    ordered_edge = tuple(sorted((parent, node)))
                    betweenness[ordered_edge] += edge_credit
                    node_credit[parent] += edge_credit
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
community_output_file_path = sys.argv[4]

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

# part1
with open(output_file_path, 'w') as file:
    for edge, betweenness_value in sorted_betweenness.collect():
        line = f"{edge}, {round(betweenness_value, 5)}\n"
        file.write(line)

# part2
def get_communities(adj_list):
    communities = []
    visited = set()

    for node in adj_list.keys():
        if node not in visited:
            visited.add(node)
            community = {node}
            queue = [node]
            while queue:
                current = queue.pop(0)
                for neighbor in adj_list.get(current, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        community.add(neighbor)
            communities.append(community)

    return communities

def modularity(communities, m, ki, original_edges):
    edges_set = {frozenset((i, j)) for i, j in original_edges}
    Q = 0.0

    for community in communities:
        for i in community:
            for j in community:
                # check if edge exists
                A_ij = 1 if frozenset((i, j)) in edges_set else 0
                # calculate modularity
                Q += (A_ij - ki[i] * ki[j] / (2.0 * m))
    Q /= (2.0 * m)
    return Q

def update_edges(adjacency_list):
    """ Rebuild edges from the updated adjacency list. """
    return [(node, neighbor) for node, neighbors in adjacency_list.items() for neighbor in neighbors]

def remove_high_betweenness_edges(current_betweenness, adjacency_list):
    """ Remove edges with the highest betweenness. """
    max_betweenness_value = max(current_betweenness.values())
    edges_to_remove = [edge for edge, bt in current_betweenness.items() if bt == max_betweenness_value]
    for edge in edges_to_remove:
        adjacency_list[edge[0]].discard(edge[1])
        adjacency_list[edge[1]].discard(edge[0])
    return adjacency_list

# m、ki
original_edges = edges.collect()
m = len(original_edges)
ki = defaultdict(int)
for edge in original_edges:
    ki[edge[0]] += 1
    ki[edge[1]] += 1

adjacency_list = edges.flatMap(lambda edge: [edge, (edge[1], edge[0])]) \
    .groupByKey().mapValues(set).collectAsMap()

# Compute initial communities and modularity
best_communities = get_communities(adjacency_list)
max_modularity = modularity(best_communities, m, ki, original_edges)

# Main loop
while edges.count() > 0:
    current_betweenness = calculate_betweenness(sc.parallelize(update_edges(adjacency_list))).collectAsMap()
    adjacency_list = remove_high_betweenness_edges(current_betweenness, adjacency_list)
    edges = sc.parallelize(update_edges(adjacency_list))

    # Detect communities with updated adjacency list
    communities = get_communities(adjacency_list)

    # Calculate modularity using the original graph for ki and kj, and the updated graph for A
    current_modularity = modularity(communities, m, ki, set(edges.collect()))
    # print("Current modularity:", current_modularity)

    # Update best communities and max modularity if modularity increased
    if current_modularity > max_modularity:
        best_communities = communities
        max_modularity = current_modularity

print(max_modularity)

# output community results
with open(community_output_file_path, 'w') as file:
    for community in sorted([sorted(list(community)) for community in best_communities], key=lambda x: (len(x), x)):
        community_with_quotes = [f"'{member}'" for member in community]
        file.write(f"{', '.join(community_with_quotes)}\n")