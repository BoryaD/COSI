from matplotlib import pyplot
from PIL import Image
import numpy as np
# from scipy.ndimage.filters import convolve


def print_hist_1_channel(arr, name):
    pyplot.hist(arr.flatten(), bins=256)
    pyplot.title(f"Histogram of grayscale image : {name}")
    pyplot.show()


def print_hist_3_channel(arr, name):
    array_1d = arr.flatten()
    pyplot.hist(array_1d[::3], bins=256)
    pyplot.title(f"R channel of image {name}")
    pyplot.show()
    pyplot.hist(array_1d[1::3], bins=256)
    pyplot.title(f"G channel of image {name}")
    pyplot.show()
    pyplot.hist(array_1d[2::3], bins=256)
    pyplot.title(f"B channel of image {name}")
    pyplot.show()


def get_pixel(img, x, y):

    if x < 0 or y < 0 or x > img.shape[0] - 1 or y > img.shape[1] -1:
        return 0

    return img[x][y]


def convolve(img, core):
    new_img = []
    for x in range(len(img)):
        new_line = []
        for y in range(len(img[0])):

            neighbors = []

            neighbors.append(get_pixel(img, x - 1, y - 1))
            neighbors.append(get_pixel(img, x, y - 1))
            neighbors.append(get_pixel(img, x + 1, y - 1))
            neighbors.append(get_pixel(img, x - 1, y))
            neighbors.append(get_pixel(img, x, y))
            neighbors.append(get_pixel(img, x + 1, y))
            neighbors.append(get_pixel(img, x - 1, y + 1))
            neighbors.append(get_pixel(img, x, y + 1))
            neighbors.append(get_pixel(img, x + 1, y + 1))
            mx = np.array(neighbors)
            res = np.multiply(mx, core.flatten())  # делает массив одномерным

            new_line.append(sum(res))

        new_img.append(np.array(new_line))
    return np.array(new_img)

f_max = 200
g_max = 90
border = 255


def dissection(arr):
    sub = border - g_max
    additive = border - f_max
    arr1 = np.where(arr > f_max, arr + additive, border)
    arr2 = np.where(arr < sub, 0, arr - sub)
    return arr1, arr2


def prewitt_filter_3_channels(arr):

    P = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    Q = np.array([[-1, 0, -1], [-1, 0, 1], [-1, 0, 1]])

    channels1 = []
    channels2 = []

    for channel in range(3):
        res1 = convolve(arr[:, :, channel], P)
        res2 = convolve(arr[:, :, channel], Q)
        channels1.append(res1)
        channels2.append(res2)

    res1 = np.dstack((channels1[0], channels1[1], channels1[2]))
    res2 = np.dstack((channels2[0], channels2[1], channels2[2]))

    # res = np.sqrt(np.square(res1) + np.square(res2))
    res = np.maximum(res1, res2)
    #
    # res = np.where(res < 0, 0, res)
    # res = np.where(res > border, border, res)

    return res



def prewitt_filter_1_channel(arr):
    P = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    Q = np.array([[-1, 0, -1], [-1, 0, 1], [-1, 0, 1]])

    # P = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    # Q = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    res1 = convolve(arr, P)
    res2 = convolve(arr, Q)

    # res = np.sqrt(np.square(res1) + np.square(res2))

    res = np.maximum(res1, res2)

    # res = np.where(res < 0, 0, res)
    # res = np.where(res > border, border, res)

    return res


def main():
    image_src_gs = Image.open('images/Lab1/me_pgm.pgm')
    data_gs = np.asarray(image_src_gs)
    image_src_rgb = Image.open('images/Lab1/me.jpg')
    data_rgb = np.asarray(image_src_rgb)

    print_hist_1_channel(data_gs, "Source")
    dis_up, dis_down = dissection(data_gs)
    dis_up_rgb, dis_down_rgb = dissection(data_rgb)

    print_hist_1_channel(dis_up, "Dissection up")
    print_hist_1_channel(dis_down, "Dissection down")

    image_dis_up = Image.fromarray(dis_up_rgb)
    image_dis_down = Image.fromarray(dis_down_rgb)
    image_dis_down.save('images/Lab1/me_pgm_down_rgb.jpg')
    image_dis_up.save('images/Lab1/me_pgm_up_rgb.jpg')


    image_dis_up = Image.fromarray(dis_up)
    image_dis_down = Image.fromarray(dis_down)
    image_dis_down.save('images/Lab1/me_pgm_down.pgm')
    image_dis_up.save('images/Lab1/me_pgm_up.pgm')



    print_hist_3_channel(data_rgb, "Source")

    filtered = prewitt_filter_3_channels(data_rgb)

    print_hist_3_channel(filtered, "Filtered RGB")

    filtered_image = Image.fromarray(filtered.astype(np.uint8))
    filtered_image.save("images/Lab1/me_filtered.jpg")

    filtered_gs = prewitt_filter_1_channel(data_gs)
    print_hist_1_channel(filtered_gs, "Filtered GS")

    filtered_image_gs = Image.fromarray(filtered_gs.astype(np.uint8))
    filtered_image_gs.save("images/Lab1/me_filtered_gs.jpg")


if __name__ == '__main__':

    main()
