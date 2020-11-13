from PIL import Image
import numpy as np
from scipy.ndimage.filters import convolve
import random


def create_clustering_image(objects, vectors, cluster_name):
    step = 255 // (len(cluster_name))
    steps = 0
    colors = np.unique(objects, return_counts=True)[0][1:]
    cluster_num = len(vectors[0]) - 1
    r = objects.copy()
    b = objects.copy()
    colors_new = [[127, 38], [150, 200], [16, 66], [20, 200], [0, 0]]

    for cl in range(len(cluster_name)):
        steps += step
        for i in range(len(vectors)):
            if vectors[i][cluster_num] == cluster_name[cl]:
                objects = np.where(objects == colors[i], steps, objects)
                r = np.where(r == colors[i], colors_new[cl][0], r)
                b = np.where(b == colors[i], colors_new[cl][1], b)
    objects = np.dstack((r, objects, b))
    return objects


def clustering(vectors, centroids):
    cluster_num = len(vectors[0]) - 1
    for vector in vectors:
        min_dist = -1
        min_ind = 0
        for centroid in centroids:
            dist = np.linalg.norm(vector[:-1] - centroid[:-1])
            if min_dist == -1 or dist < min_dist:
                min_dist = dist
                min_ind = centroid[cluster_num]
        vector[cluster_num] = min_ind
    return vectors


def find_new_centroids(vectors, centroids):
    cluster_num = len(vectors[0]) - 1
    new_centroids = []
    for centroid in centroids:
        counter = 0
        sum_ = np.zeros(cluster_num + 1)
        for vector in vectors:
            if vector[cluster_num] == centroid[cluster_num]:
                counter += 1
                sum_ = vector + sum_
        sum_ /= counter
        new_centroids.append(sum_)
    return np.array(new_centroids)


def mk_means(vectors, k):
    cluster_num = len(vectors[0]) - 1
    centroids_indexes = random.sample(range(len(vectors)-1), k)
    centroids = []
    for i in range(len(centroids_indexes)):
        vectors[centroids_indexes[i]][cluster_num] = i
        centroids.append(vectors[centroids_indexes[i]])
    centroids = np.array(centroids)
    vectors = clustering(vectors, centroids)

    while True:
        new_centroids = find_new_centroids(vectors, centroids)
        vectors = clustering(vectors, new_centroids)
        # print(centroids)
        # print(new_centroids)
        # print("")
        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids

    return vectors, centroids


def find_moments(array, mass_center):
    moments = []
    arr = array.copy()
    colors = np.unique(arr, return_counts=True)[0][1:]
    for i in range(len(mass_center)):
        pixels = np.where(arr == colors[i])
        xs = pixels[0] - mass_center[i][0]
        ys = pixels[1] - mass_center[i][1]
        m20 = sum(np.square(xs))
        m02 = sum(np.square(ys))
        m11 = sum(xs * ys)
        moments.append([m20, m02, m11])
    return moments


def find_eccentricity(moments):
    res = []
    for i in range(len(moments)):
        for j in range(len(moments[i])):
            moments[i][j] = int(moments[i][j])

    for moment in moments:
        summ = moment[0] + moment[1]
        sub = moment[0] - moment[1]

        c = (summ + np.sqrt(summ ** 2 + 4 * moment[2] ** 2))
        z = (sub + np.sqrt(summ ** 2 + 4 * moment[2] ** 2))
        if z == 0:
            z = 0.00000000000000000000000000000000000000000000000000001

        r = c/z
        res.append(r)
    return res


def perimeter_founder(arr):
    perimeter = arr.copy()
    for i in range(len(arr[1:-1])):
        for j in range(len(arr[i][1:-1])):
            if arr[i][j] != 0 and arr[i][j] == arr[i - 1][j] == arr[i][j - 1] == arr[i + 1][j] == arr[i][j + 1]:
                perimeter[i][j] = 0
    return perimeter


def find_mass_center(array, squares):
    arr = array.copy()
    colors = np.unique(arr, return_counts=True)[0][1:]
    res = []
    for i in range(len(colors)):
        count = np.where(arr == colors[i])
        res.append([sum(count[0])//squares[i], sum(count[1])//squares[i]])
    for elem in res:
        arr[elem[0]][elem[1]] = 0
    return res, arr


def item_detector(arr):
    step = 11
    for i in range(len(arr[1:-1])):
        for j in range(len(arr[i][1:-1])):
            A = arr[i][j]
            B = arr[i][j - 1]
            C = arr[i - 1][j]
            if A == 0:
                continue
            elif B == C == 0:
                arr[i][j] = step

                step += 30
            elif B != 0 and C == 0:
                arr[i][j] = B
            elif C != 0 and B == 0:
                arr[i][j] = C
            elif B != 0 and C != 0:
                if B == C:
                    arr[i][j] = C
                else:
                    arr[i][j] = B
                    arr = np.where(arr == C, B, arr)

    new_colors = np.unique(arr)
    step = 255 // (len(new_colors) - 1)
    steps = 0
    for elem in new_colors[1:]:
        steps += step
        arr = np.where(arr == elem, steps, arr)
    return arr


def filer(arr):
    P = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
    res = convolve(arr, P)
    res = np.where(res < 0, 0, res)
    res = np.where(res > 255, 255, res)
    return res


def to_binary(arr, threshold):
    return np.where(arr <= threshold, 0, 255)


def main():
    num_image = 4
    # image = Image.open(f'images/Lab2/{num_image}e.jpg').convert('L')
    # data = np.asarray(image).astype(np.int16)
    # data = filer(data)
    # data = filer(data)
    # data = filer(data)
    # data = filer(data)
    #
    # data = to_binary(data, 180)
    # Image.fromarray(data.astype(np.uint8)).save(f'images/Lab2/res{num_image}e.pgm')

    image = Image.open(f'images/Lab2/res{num_image}e.pgm')
    data = np.asarray(image).astype(np.int64)
    objects = item_detector(data)
    Image.fromarray(objects.astype(np.uint8)).save(f'images/Lab2/objects{num_image}e.pgm')
    squares = np.unique(objects, return_counts=True)[1][1:]
    perimeters = perimeter_founder(objects)
    Image.fromarray(perimeters.astype(np.uint8)).save(f'images/Lab2/perimeters{num_image}e.pgm')
    perimeters = np.unique(perimeters, return_counts=True)[1][1:]
    compactness = np.square(perimeters) / squares
    mass_center, msc = find_mass_center(objects, squares)
    Image.fromarray(msc.astype(np.uint8)).save(f'images/Lab2/mass_centers{num_image}e.pgm')
    moments = find_moments(objects, mass_center)
    eccentricity = find_eccentricity(moments)
    vectors = np.dstack((squares, perimeters, compactness, eccentricity, np.full(len(squares), -1)))
    vectors = vectors[0]
    # from sklearn.cluster import KMeans
    # vectors2 = np.dstack((squares, perimeters, eccentricity))
    # vectors2 = vectors2[0]
    # X = vectors2
    #
    # clusterNum = 4
    # k_means = KMeans(n_clusters=clusterNum)
    # k_means.fit(X)  # Compute k-means clustering
    # labels = k_means.labels_
    # print(labels)
    vectors, clusters = mk_means(vectors, 5)
    cluster_name = clusters.transpose()[len(clusters[0]) - 1]

    clustering_objects = create_clustering_image(objects, vectors, cluster_name)
    Image.fromarray(clustering_objects.astype(np.uint8)).save(f'images/Lab2/clustered{num_image}e.jpg')


if __name__ == '__main__':

    main()
