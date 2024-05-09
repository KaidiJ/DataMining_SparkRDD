import sys
from sklearn.cluster import KMeans
import numpy as np

def mahalanobis_distance(x, mean, std):
    if np.any(std == 0):
        std = np.where(std == 0, 1, std)
    return np.sqrt(np.sum((x - mean) ** 2 / std ** 2))

def find_closest_cluster(data_point, cluster_centers, cluster_variances):
    min_distance = float('inf')
    closest_cluster = None
    for cluster_id, center in cluster_centers.items():
        std_dev = cluster_variances[cluster_id]
        distance = mahalanobis_distance(data_point, center, std_dev)
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster_id
    return closest_cluster, min_distance
def main():
    data_path = sys.argv[1]
    num_clusters = int(sys.argv[2])
    save_path = sys.argv[3]

    # Temporary storage for data
    outlier_indexes = set()
    detailed_stats = {}
    cluster_centers = {}
    cluster_variances = {}
    points_per_cluster = {}
    small_cluster_stats = {}
    small_cluster_centers = {}
    small_cluster_variances = {}
    small_points_per_cluster = {}
    result = {}

    max_distance = 3

    with open(data_path, "r") as file:
        raw_data = np.array([line.strip().split(',') for line in file.readlines()], dtype=np.float64)
        total_data_points = raw_data.shape[0]

    # Step 1: Random shuffle
    np.random.shuffle(raw_data)
    split_data = np.array_split(raw_data, 5)
    first_split = split_data[0]
    processed = np.zeros(len(raw_data), dtype=bool)  # Track processed data here

    with open(save_path, "w") as file:  # Open file here and keep it open until all writing is done
        file.write('The intermediate results:\n')

        # Step 2: Initial clustering
        initial_kmeans = KMeans(n_clusters=5 * num_clusters).fit(first_split[:, 2:])

        # Step 3: Outlier detection
        label_clusters = {}
        for index, cluster_id in enumerate(initial_kmeans.labels_):
            if cluster_id not in label_clusters:
                label_clusters[cluster_id] = []
            label_clusters[cluster_id].append(index)

        for indices in label_clusters.values():
            if len(indices) == 1:
                outlier_indexes.add(indices[0])
        non_outliers = np.delete(first_split, list(outlier_indexes), axis=0)

        # Step 4: Secondary clustering
        secondary_kmeans = KMeans(n_clusters=num_clusters).fit(non_outliers[:, 2:])
        label_clusters = {}
        for index, cluster_id in enumerate(secondary_kmeans.labels_):
            if cluster_id not in label_clusters:
                label_clusters[cluster_id] = []
            label_clusters[cluster_id].append(index)

        # Step 5: Computing stats
        for cluster_id, indices in label_clusters.items():
            cluster_data = non_outliers[indices, 2:]
            cluster_sum = np.sum(cluster_data, axis=0)
            cluster_sumsq = np.sum(np.square(cluster_data), axis=0)
            count = len(indices)
            detailed_stats[cluster_id] = [count, cluster_sum, cluster_sumsq]
            points = non_outliers[indices, 0].astype(int).tolist()
            points_per_cluster[cluster_id] = points
            cluster_center = cluster_sum / count
            cluster_centers[cluster_id] = cluster_center
            cluster_variances[cluster_id] = np.sqrt(np.subtract(cluster_sumsq / count, np.square(cluster_center)))
            for point in points:
                result[point] = cluster_id

            # Mark processed data
            processed = np.zeros(len(raw_data), dtype=bool)

            # initialization for the remaining chunks loop
            cumulative_ds = sum([value[0] for value in detailed_stats.values()])
            cumulative_cs = sum([value[0] for value in small_cluster_stats.values()])
            cumulative_rs = len(outlier_indexes)

            # Output results of Round 1
            result_str = f'Round 1: {cumulative_ds},{len(small_cluster_stats)},{cumulative_cs},{cumulative_rs}\n'
            file.write(result_str)

        for i in range(1, 5):
            current_data = split_data[i]
            rs = set()
            for point_idx, point in enumerate(current_data):
                if not processed[point_idx]:  # Check if the point has been processed
                    data_point = point[2:]
                    closest_cluster, min_distance = find_closest_cluster(data_point, cluster_centers, cluster_variances)

                    if min_distance < max_distance and closest_cluster is not None:
                        stats = detailed_stats[closest_cluster]
                        new_count = stats[0] + 1
                        new_sum = stats[1] + data_point
                        new_sumsq = stats[2] + data_point ** 2
                        detailed_stats[closest_cluster] = [new_count, new_sum, new_sumsq]
                        points_per_cluster[closest_cluster].append(int(point[0]))
                        processed[point_idx] = True
                        points_per_cluster[closest_cluster].append(point_idx)
                    else:
                        rs.add(point_idx)

            # Handling the outliers by re-clustering them
            if len(outlier_indexes) > 0:
                if len(outlier_indexes) >= 5 * num_clusters:
                    outlier_data = current_data[list(outlier_indexes), :]
                    if outlier_data.shape[0] >= 50:
                        recluster_kmeans = KMeans(n_clusters=max(5 * num_clusters, outlier_data.shape[0])).fit(outlier_data[:, 2:])
                        outlier_clusters = {}
                        if 'recluster_kmeans' in locals():
                            for idx, label in enumerate(recluster_kmeans.labels_):
                                if label not in outlier_clusters:
                                    outlier_clusters[label] = []
                                outlier_clusters[label].append(idx)
                        outlier_indexes = set()
                        for indices in outlier_clusters.values():
                            if len(indices) == 1:
                                outlier_indexes.add(indices[0])

                # Reevaluating clusters to potentially merge based on their proximity
                for cid1 in list(small_cluster_stats.keys()):
                    for cid2 in list(small_cluster_stats.keys()):
                        if cid1 != cid2:
                            centroid1 = small_cluster_centers[cid1]
                            centroid2 = small_cluster_centers[cid2]
                            std_dev1 = small_cluster_variances[cid1]
                            std_dev2 = small_cluster_variances[cid2]
                            distance = np.sqrt(np.sum((centroid1 - centroid2) ** 2))
                            if distance < max_distance:  # Threshold for merging
                                # Merge clusters
                                merged_count = small_cluster_stats[cid1][0] + small_cluster_stats[cid2][0]
                                merged_sum = small_cluster_stats[cid1][1] + small_cluster_stats[cid2][1]
                                merged_sumsq = small_cluster_stats[cid1][2] + small_cluster_stats[cid2][2]
                                small_cluster_stats[cid2] = [merged_count, merged_sum, merged_sumsq]
                                small_cluster_centers[cid2] = merged_sum / merged_count
                                small_cluster_variances[cid2] = np.sqrt((merged_sumsq / merged_count) - (merged_sum / merged_count) ** 2)
                                small_points_per_cluster[cid2].extend(small_points_per_cluster[cid1])

                                del small_cluster_stats[cid1]
                                del small_cluster_centers[cid1]
                                del small_cluster_variances[cid1]
                                del small_points_per_cluster[cid1]

            # Handling the outliers and merging clusters in the last data split
            if i == 4:  # Last data split, index starts from 0 hence 4
                # Step 13: Re-clustering outliers if necessary
                if len(outlier_indexes) > 0:
                    if len(outlier_indexes) >= 5 * num_clusters:
                        outlier_data = current_data[list(outlier_indexes), :]
                        if outlier_data.shape[0] >= 50:
                            recluster_kmeans = KMeans(n_clusters=max(5 * num_clusters, outlier_data.shape[0])).fit(
                                outlier_data[:, 2:])
                            outlier_clusters = {}
                            if 'recluster_kmeans' in locals():
                                for idx, label in enumerate(recluster_kmeans.labels_):
                                    if label not in outlier_clusters:
                                        outlier_clusters[label] = []
                                    outlier_clusters[label].append(idx)
                            outlier_indexes = set()
                            for indices in outlier_clusters.values():
                                if len(indices) == 1:
                                    outlier_indexes.add(indices[0])

                # Step14 Merging small clusters with similar centroids
                for cid1 in list(small_cluster_stats.keys()):
                    for cid2 in list(detailed_stats.keys()):
                        centroid1 = small_cluster_centers[cid1]
                        centroid2 = cluster_centers[cid2]
                        std_dev1 = small_cluster_variances[cid1]
                        std_dev2 = cluster_variances[cid2]
                        distance = np.sqrt(np.sum((centroid1 - centroid2) ** 2))
                        if distance < max_distance:  # Threshold for merging
                            # Merge clusters
                            merged_count = small_cluster_stats[cid1][0] + detailed_stats[cid2][0]
                            merged_sum = small_cluster_stats[cid1][1] + detailed_stats[cid2][1]
                            merged_sumsq = small_cluster_stats[cid1][2] + detailed_stats[cid2][2]
                            detailed_stats[cid2] = [merged_count, merged_sum, merged_sumsq]
                            cluster_centers[cid2] = merged_sum / merged_count
                            cluster_variances[cid2] = np.sqrt(
                                (merged_sumsq / merged_count) - (merged_sum / merged_count) ** 2)
                            points_per_cluster[cid2].extend(small_points_per_cluster[cid1])

                            del small_cluster_stats[cid1]
                            del small_cluster_centers[cid1]
                            del small_cluster_variances[cid1]
                            del small_points_per_cluster[cid1]
            # Write intermediate results to file
            # Calculate cumulative values here if needed
            cumulative_ds = sum([value[0] for value in detailed_stats.values()])
            num_cs = sum([value[0] for value in points_per_cluster.values()])
            result_str = f'Round {i}: {cumulative_ds},{len(small_cluster_stats)},{cumulative_cs},{cumulative_rs}\n'
            file.write(result_str)

        # Writing clustering results to file
        file.write('\nThe clustering results:\n')
        for point_idx in range(len(raw_data)):
            if processed[point_idx]:
                cluster_id = points_per_cluster.get(point_idx, -1)
                file.write(f"{point_idx}, {cluster_id}\n")
            else:
                file.write(f"{point_idx}, -1\n")
if __name__ == '__main__':
    main()
