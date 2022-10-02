#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Author(s): 
Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sklearn.cluster
import skimage.transform as nimage


def get_pixel(im, j, i, c):
    # Padding
    row, column, channel = im.shape
    if i < 0:
        i = 0
    if i > row-1:
        i = row-1
    if j < 0:
        j = 0
    if j > column-1:
        j = column-1

    pixel = im[i, j, c]
    return pixel


def set_pixel(im, j, i, c, value):
    im[i, j, c] = value

# Following is a function to construct gaussian filter, with scale sigma


def make_gaussian_filter(size, sigma):
    #  We need an odd dimentional array
    if size % 2 == 0:
        size += 1

     # Single Channel array of floats.
    gaussian = np.zeros(shape=(size, size, 1))
    for i in range(size):
        for j in range(size):
            x = j - (size - 1) / 2
            y = i - (size - 1) / 2
            value = (1 / (2 * np.pi * sigma*sigma)) * \
                np.exp(-(x*x + y*y) / (2 * sigma*sigma))
            set_pixel(gaussian, i, j, 0, value.real)
    return gaussian


def make_gaussian2d_filter(size, sigma):
    #  We need an odd dimentional array
    if size % 2 == 0:
        size += 1
    sigma_x = sigma
    sigma_y = 3*sigma
    # Single Channel array of floats.
    gaussian = np.zeros((size, size, 1))
    for i in range(size):
        for j in range(size):
            x = j - (size - 1) / 2
            y = i - (size - 1) / 2
            value = (1 / (2 * np.pi * sigma*sigma)) * \
                np.exp(-((x*x / (2 * sigma_x*sigma_x)) +
                       (y*y / (2 * sigma_y*sigma_y))))
            set_pixel(gaussian, i, j, 0, value.real)
            # gaussian[i,j,0] = value
    return gaussian


def apply_filter(im, filter, i, j, channel):
    f_r, f_c, f_chan = filter.shape
    sum = 0
    mid_c = (f_c+1)/2
    mid_r = (f_r+1)/2
    for a in range(0, f_r):
        for b in range(0, f_c):
            value = get_pixel(im, int(b + j - mid_c + 1), int(a + i -
                              mid_r + 1), channel) * get_pixel(filter, b, a, 0)
            sum += value
    return sum


def convolve(im, filter):
    im_row, im_column, im_channel = im.shape
    f_row, f_column, f_channel = filter.shape
    new = np.zeros(shape=(im_row, im_column, im_channel),
                   dtype=float, order='F')
    for channel in range(im_channel):
        for i in range(im_row):
            for j in range(im_column):
                val = apply_filter(im, filter, i, j, channel)
                set_pixel(new, j, i, channel, val)

    return new


# Generating A Sobel Filter
def sobel_kernal():
    gx = np.zeros(shape=(3, 3, 1))
    set_pixel(gx, 0, 0, 0, -1)
    set_pixel(gx, 0, 1, 0, -2)
    set_pixel(gx, 0, 2, 0, -1)
    set_pixel(gx, 1, 0, 0, 0)
    set_pixel(gx, 1, 1, 0, 0)
    set_pixel(gx, 1, 2, 0, 0)
    set_pixel(gx, 2, 0, 0, 1)
    set_pixel(gx, 2, 1, 0, 2)
    set_pixel(gx, 2, 2, 0, 1)
    return gx

# Function for derivative in X direfction


def x_derivative_of_image(image):
    derivative = np.zeros(shape=(3, 1, 1),
                          dtype=float, order='F')
    set_pixel(derivative, 0, 0, 0, -1)
    set_pixel(derivative, 0, 1, 0, 0)
    set_pixel(derivative, 0, 2, 0, 1)
    new = convolve(image, derivative)
    return new

# Function for derivative in Y direfction


def y_derivative_of_image(image):
    derivative = np.zeros(shape=(1, 3, 1),
                          dtype=float, order='F')
    set_pixel(derivative, 0, 0, 0, -1)
    set_pixel(derivative, 1, 0, 0, 0)
    set_pixel(derivative, 2, 0, 0, 1)
    new = convolve(image, derivative)
    return new


def laplacian(image):
    d_x = x_derivative_of_image(image)
    dd_x = x_derivative_of_image(d_x)
    d_y = y_derivative_of_image(image)
    dd_y = y_derivative_of_image(d_y)
    final = dd_x + dd_y
    return final


def DOGFilters(scales, rotations, size):
    kernels = []
    for sigma in scales:
        orients = np.linspace(0, 360, rotations)
        gauss = make_gaussian_filter(size, sigma)
        diffrentiate = sobel_kernal()
        sobel = convolve(gauss, diffrentiate)
        for i, orient in enumerate(orients):
            image = nimage.rotate(sobel, orient)
            kernels.append(image)
            image = 0
    return kernels


def Gabor_filters(scales, rotations, Lambda):
    kernels = []
    for sigma in scales:
        orients = np.linspace(0, 360, rotations)
        gabor2 = gabor(sigma, np.pi/4, Lambda, 1, 1)
        for i, orient in enumerate(orients):
            image = nimage.rotate(gabor2, orient)
            kernels.append(image)
            image = 0
    return kernels


def lm_filters(scales, rotation, size):
    kernels = []
    new_scales = scales.copy()
    new_scales.pop(len(new_scales)-1)
    for sigma in new_scales:
        orients = np.linspace(0, 360, rotation)
        gauss2d = make_gaussian2d_filter(size, sigma)
        diff_of_gauss = x_derivative_of_image(gauss2d)
        second_diff_gauss = x_derivative_of_image(diff_of_gauss)
        for i, orient in enumerate(orients):
            image = nimage.rotate(diff_of_gauss, orient)
            kernels.append(image)
            image = 0
        for i, orient in enumerate(orients):
            image = nimage.rotate(second_diff_gauss, orient)
            kernels.append(image)
            image = 0

    for sigma in scales:
        gauss = make_gaussian_filter(size, sigma)
        image = laplacian(gauss)
        kernels.append(image)
        image = 0

    for sigma in scales:
        sigma = 3*sigma
        gauss = make_gaussian_filter(size, sigma)
        image = laplacian(gauss)
        kernels.append(image)
        image = 0

    for sigma in scales:
        gauss = make_gaussian_filter(size, sigma)
        kernels.append(gauss)
        image = 0
    return kernels


def lm(filters, path):
    num = len(filters)
    plt.subplots(int(num/5), 5, figsize=(12, 4))
    for i in range(num):
        plt.subplot(int(num/12), 12, i+1)
        plt.axis('off')
        plt.imshow(filters[i], cmap='gray')
    plt.savefig(path+'Code/Solutions/LM.png')
    plt.close()


def DoG(filters, path):
    num = len(filters)
    plt.subplots(int(num/10), 10, figsize=(16, 3))
    for i in range(num):
        plt.subplot(int(num/16), 16, i+1)
        plt.axis('off')
        plt.imshow(filters[i], cmap='gray')
    plt.savefig(path+'Code/Solutions/DoG.png')
    plt.close()


def plt_gabor(filters, path):
    num = len(filters)
    plt.subplots(int(num/5), 5, figsize=(12, 5))
    for i in range(num):
        plt.subplot(int(num/12), 12, i+1)
        plt.axis('off')
        plt.imshow(filters[i], cmap='gray')
    plt.savefig(path+'Code/Solutions/Gabor.png')
    plt.close()


def gabor(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3  # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)),
               abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)),
               abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 /
                sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    gb
    return gb


# Staacking different resposes
def stack(image, filter_bank):
    m, n, c = image.shape
    out = np.zeros(shape=(m, n, c))
    for i in range(len(filter_bank)):
        new = cv.filter2D(image, -1, filter_bank[i])
        out = np.dstack((out, new))
    return out


def Texton(image, bins, dog, lms, lml, gabor):
    p, q, _ = image.shape
    dog_stack = stack(image, dog)
    lms_stack = stack(image, lms)
    lml_stack = stack(image, lml)
    gabor_stack = stack(image, gabor)
    final = np.dstack(
        (dog_stack[:, :, 1:], lms_stack[:, :, 1:], lml_stack[:, :, 1:], gabor_stack[:, :, 1:]))
    _, _, r = final.shape
    temp = np.reshape(final, ((p*q), r))
    k_means = sklearn.cluster.KMeans(n_clusters=bins)
    k_means.fit(temp)
    labels = k_means.predict(temp)
    l = np.reshape(labels, (p, q))
    return l


def brighten(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    r, c = gray.shape
    gray2 = np.reshape(gray, ((r*c), 1))
    k_means = sklearn.cluster.KMeans(n_clusters=16)
    k_means.fit(gray2)
    labels = k_means.predict(gray2)
    l = np.reshape(labels, (r, c))
    return l


def color_map(image):
    r, c, ch = image.shape
    gray2 = np.reshape(image, ((r*c), ch))
    k_means = sklearn.cluster.KMeans(n_clusters=16)
    k_means.fit(gray2)
    labels = k_means.predict(gray2)
    l = np.reshape(labels, (r, c))

    return l


def half_disc(scale):
    disk = np.zeros((scale, scale))
    if scale % 2 == 0:
        mid = scale//2
    else:
        mid = (scale-1)//2
    for i in range(mid):
        for j in range(scale):
            if np.sqrt(((i-mid)**2)+((j-mid)**2)) <= mid:
                disk[i, j] = 1
            else:
                disk[i, j] = 0

    return disk


def plt_disc(filters, path):
    num = len(filters)
    plt.subplots(int(num/5), 5, figsize=(36, 3))
    for i in range(num):
        plt.subplot(int(num/36), 36, i+1)
        plt.axis('off')
        plt.imshow(filters[i], cmap='gray')
    plt.savefig(path+'Code/Solutions/HDMasks.png')
    plt.close()


def disk_filters(scales, rotations):
    kernels = []
    for scale in scales:
        orients = np.linspace(0, 180, rotations)
        gabor2 = half_disc(scale)
        for i, orient in enumerate(orients):
            image = nimage.rotate(gabor2, orient)
            kernels.append(image)
            image = nimage.rotate(gabor2, orient+180)
            kernels.append(image)
            image = 0
    return kernels


def gradient(image, bins, filter_bank):
    temp = image
    for i in range(len(filter_bank)//2):
        g = chi_square(image, bins, filter_bank[2*i], filter_bank[2*i+1])
        temp = np.dstack((temp, g))
    mean = np.mean(temp, axis=2)
    return mean


def chi_square(image, bins, kernal1, kernal2):
    r, c = image.shape
    temp = np.zeros(shape=(r, c))
    chi = np.zeros(shape=(r, c))
    for i in range(bins):
        temp[image == i] = 1
        g = cv.filter2D(temp, -1, kernal1)
        h = cv.filter2D(temp, -1, kernal2)
        if (g+h != 0).all() == True:
            chi = chi + ((g-h)**2)/(g+h)
    return chi


# Function that returns a half-disk filter bank


def main():

    # Change path here
    #################
    path = '/home/naveen/Desktop/nmangla_hw0/nmangla_hw0/Phase1/'
    #################

    filter_dog = DOGFilters([5, 9, 15], 16, 65)

    filter_lms = lm_filters([np.sqrt(2), 2, 2*np.sqrt(2), 4], 6, 49)
    filter_lml = lm_filters([3*np.sqrt(2), 3*2, 3*2*np.sqrt(2), 3*4], 6, 49)
    lm(filter_lms, path)  # Saving LM.png

    filter_gabor = Gabor_filters([5, 9, 15], 12, 2)
    plt_gabor(filter_gabor, path)  # Saving Gabor.png

    disk = disk_filters([15, 25, 35], 18)
    plt_disc(disk, path)  # Disks

    for i in range(10):
        print("Working on "+str(i+1)+".png")
        image = cv.imread(path+'BSDS500/Images/'+str(i+1)+'.jpg', 1)
        l = Texton(image, 64, filter_dog, filter_lms, filter_lml, filter_gabor)
        np.save(path+'Code/Solutions/'+str(i+1)+'/NewT_'+str(i+1), l)
        print("Texture map saved for "+str(i+1)+".png")
        l1 = brighten(image)
        np.save(path+'Code/Solutions/'+str(i+1)+'/NewB_'+str(i+1), l1)
        l2 = color_map(image)
        np.save(path+'Code/Solutions/'+str(i+1)+'/NewC_'+str(i+1), l2)

        T = np.load(path+'Code/Solutions/'+str(i+1)+'/NewT_'+str(i+1)+'.npy')
        B = np.load(path+'Code/Solutions/'+str(i+1)+'/NewB_'+str(i+1)+'.npy')
        C = np.load(path+'Code/Solutions/'+str(i+1)+'/NewC_'+str(i+1)+'.npy')

        plt.imsave(path+'Code/Solutions/'+str(i+1) +
                   '/Texton_map_'+str(i+1)+'.png', T)
        plt.imsave(path+'Code/Solutions/'+str(i+1) +
                   '/Brightness_map_'+str(i+1)+'.png', B)
        plt.imsave(path+'Code/Solutions/'+str(i+1) +
                   '/Color_map_'+str(i+1)+'.png', C)

        T_g = gradient(T, 64, disk)
        plt.imsave(path+'Code/Solutions/'+str(i+1) +
                   '/Texton_gradient_'+str(i+1)+'.png', T_g)
        B_g = gradient(B, 16, disk)
        plt.imsave(path+'Code/Solutions/'+str(i+1) +
                   '/Brightness_gradient_'+str(i+1)+'.png', B_g)

        C_g = gradient(C, 16, disk)
        plt.imsave(path+'Code/Solutions/'+str(i+1) +
                   '/Color_gradient_'+str(i+1)+'.png', C_g)

        avg = (T_g+B_g+C_g)/3

        path_sobel = path+'BSDS500/SobelBaseline/'+str(i+1)+'.png'
        path_canny = path+'BSDS500/CannyBaseline/'+str(i+1)+'.png'
        sobel = plt.imread(path_sobel, 0)
        canny = plt.imread(path_canny, 0)
        pblite = np.multiply(avg, (0.5*canny+0.5*sobel))

        cv.imwrite(path+'Code/Solutions/'+str(i+1) +
                   '/Pb_lite_'+str(i+1)+'.png', pblite)
        print("Pb-lite saved for "+str(i+1)+".png")


if __name__ == '__main__':
    main()
