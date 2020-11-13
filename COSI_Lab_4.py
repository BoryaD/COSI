import numpy as np
import random


np.set_printoptions(suppress=True)
D = 0.2

image1 = [
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
]

image2 = [
    [1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
]

image3 = [
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0],
]

image4 = [
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0],
]

image5 = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0],
]

images = [np.array(image1).reshape(1, 49), np.array(image2).reshape(1, 49),
          np.array(image3).reshape(1, 49), np.array(image4).reshape(1, 49),
          np.array(image5).reshape(1, 49)]

y_right = np.zeros(5)

first_layer_count = 49
xs = np.zeros(first_layer_count)

second_layer_count = 7
gs = np.zeros(second_layer_count)

third_layer_count = 5
ys = np.zeros(third_layer_count)


v = np.random.uniform(low=-1, high=1, size=(first_layer_count, second_layer_count))

w = np.random.uniform(low=-1, high=1, size=(second_layer_count, third_layer_count))


Q = np.random.uniform(low=-1, high=1, size=(second_layer_count,))

T = np.random.uniform(low=-1, high=1, size=(third_layer_count,))


def print_matrix(mx):
    mx = np.where(mx == 0, " ", 0).reshape(7, 7)
    for i in mx:
        for j in i:
            print(j, end=' ')
        print()


def create_noise(per, image):
    im = image.copy().flatten()
    noise_indexes = random.sample(range(len(im)), int(per*0.49))

    for i in noise_indexes:
        if im[i] == 0:
            im[i] = 1
        else:
            im[i] = 0
    return im


def func(x):
    return 1 / (1 + np.e ** (-x))


def teach_weights(max_count, a, b):
    global gs
    global xs
    global ys
    global v
    global w
    global y_right
    global T
    global Q
    k = 0
    is_ready = 0

    for i in range(max_count):
        percent_of_noise = random.randint(0, 50)
        xs = create_noise(percent_of_noise, images[k])
        gs = xs.dot(v)
        gs += Q

        gs = func(gs)
        ys = gs.dot(w)
        ys += T
        ys = func(ys)
        y_right[k] = 1
        err = a * ys * (1-ys) * (y_right - ys)
        dw = gs.reshape(second_layer_count, 1).dot(err.reshape(1, third_layer_count))
        n_w = w + dw
        T = T + err

        err2 = np.matmul(w, err)
        mul = b * gs * (1 - gs) * err2
        v = v + (xs.reshape(first_layer_count, 1).dot(mul.reshape(1, second_layer_count)))
        Q = Q + mul

        w = n_w

        y_right[k] = 0
        k += 1
        if k == 5:
            k = 0

        if abs(sum(y_right - ys)) < 0.3:
            is_ready += 1
            # print("ready!!!")
            # break
        else:
            is_ready = 0

        if is_ready == 2:
            print("ready!!!")
            break


def find_image(im):
    xs = im.flatten()
    gs = xs.dot(v)
    gs += Q
    gs = func(gs)
    ys = gs.dot(w)
    ys += T
    ys = func(ys)
    return np.argmax(ys)


def main():
    teach_weights(1_000_000, 1, 1)

    for j in range(len(images)):
        for i in range(0, 50, 10):
            noise = create_noise(i, images[j])
            im = find_image(noise)
            print_matrix(noise)
            print("-" * 128)
            print_matrix(images[int(im)])
            print("" * 128)
            print_matrix(images[j])
            print("-" * 128)
            print("-" * 128)



if __name__ == '__main__':
    main()
