import cv2
import numpy as np
def generate_gaussian_kernal(n, sd):
    gaussian_kernal = np.zeros((n,n) , dtype=np.float32)
    factor = 1 / (2 * np.pi * sd**2)
    sum_val = 0
    for i in range(-n//2 , n//2 + 1):
        for j in range(-n//2, n//2 + 1):
            x = i + n//2
            y = j + n//2
            gaussian_kernal[x][y] = factor * np.exp(-(i**2 + j**2)/2*(sd**2))
            sum_val += gaussian_kernal[x][y]

    return gaussian_kernal / sum_val

if __name__=="__main__":
    image = cv2.imread('/home/sathvik/PycharmProjects/opencv/Images/lena.jpeg', 0)
    gaussian_kernel = generate_gaussian_kernal(5, 1)

    gaussian_filtered_image = cv2.filter2D(src = image, ddepth=-1, kernel = gaussian_kernel)
    cv2.imshow('original image', image)
    cv2.waitKey(0)
    cv2.imshow('gaussian filtered image', gaussian_filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
