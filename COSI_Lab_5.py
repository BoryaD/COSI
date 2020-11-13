import numpy as np
import random

np.set_printoptions(suppress=True)
D = 0.2
b = 0.5


image1 = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
]

image2 = [
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
]

image3 = [
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0],
]

image4 = [
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0],
]

image5 = [
    [0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 0],
]

images = [np.array(image1).reshape(1, 36), np.array(image2).reshape(1, 36),
          np.array(image3).reshape(1, 36), np.array(image4).reshape(1, 36),
          np.array(image5).reshape(1, 36)]


first_layer_count = 36
xs = np.zeros(first_layer_count)

second_layer_count = 5
ys = np.zeros(second_layer_count)

w = np.random.uniform(low=0, high=1, size=(first_layer_count, second_layer_count))


def create_noise(per, image):
    im = image.copy().flatten()
    noise_indexes = random.sample(range(len(im)), int(per*0.36))

    for i in noise_indexes:
        if im[i] == 0:
            im[i] = 1
        else:
            im[i] = 0
    return im


def print_matrix(mx):
    mx = np.where(mx == 0, " ", 0).reshape(6, 6)
    for i in mx:
        for j in i:
            print(j, end=' ')
        print()


def teach_weights(max_count):
    global xs
    global ys
    global w

    k = 0
    is_ready = 0

    for m in range(max_count):
        percent_of_noise = random.randint(0, 10)
        xs = create_noise(percent_of_noise, images[k])
        ys = xs.dot(w)
        winner = np.argmax(ys)
        dw = w[:, winner] + b * (xs - w[:, winner])
        w[:, winner] = dw/np.linalg.norm(dw)

        k += 1
        if k == 5:
            k = 0


def find_image(im):
    xs = im.flatten()
    ys = xs.dot(w)
    return np.argmax(ys)


def main():
    clusters = [[], [], [], [], []]
    teach_weights(5000)
    for j in range(len(images)):
        for i in range(0, 70, 5):
            noise = create_noise(i, images[j])
            noise = noise/np.linalg.norm(noise)
            im = find_image(noise)
            clusters[j].append(im)
            # print_matrix(noise)
            # print("-" * 128)
            # print_matrix(images[int(im)])
            # print("" * 128)
            # print_matrix(images[j])
            # print("-" * 128)
            # print("-" * 128)
    for cluster in clusters:
        print(f"Cluster: {cluster}")


if __name__ == '__main__':
    main()
