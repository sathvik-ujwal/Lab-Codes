import cv2
import numpy as np

image = cv2.imread('/home/sathvik/PycharmProjects/opencv/Images/lena.jpeg', 0)
cv2.imshow('original image', image)
cv2.waitKey(0)

horizontal_sobel_kernel = np.array([[-2, 0, 2],
                                    [-4, 0, 4],
                                    [-2, 0, 2]])

vertical_sobel_kernel = np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]])

horizontal_laplacian_kernel = np.array([[-1, -1, -1],
                                    [2, 2, 2],
                                    [-1, -1, -1]])

vertical_laplacian_kernel = np.array([[-1, 2, -1],
                                    [-1, 2, -1],
                                    [-1, 2, -1]])

horizontal_edges = cv2.filter2D(src=image, ddepth=-1, kernel=horizontal_sobel_kernel)
vertical_edges= cv2.filter2D(src=image, ddepth=-1, kernel=vertical_sobel_kernel)
horizontal_laplacian_edges = cv2.filter2D(src=image, ddepth=-1, kernel=horizontal_laplacian_kernel)
vertical_laplacian_edges= cv2.filter2D(src=image, ddepth=-1, kernel=vertical_laplacian_kernel)

combined_sobel_edges = horizontal_edges + vertical_edges

cv2.imshow('horizontal edges', horizontal_edges)
cv2.waitKey(0)
cv2.imshow('vertical edges', vertical_edges)
cv2.waitKey(0)
cv2.imshow('horizontal laplacian edges', horizontal_laplacian_edges)
cv2.waitKey(0)
cv2.imshow('vertical laplacian edges', vertical_laplacian_edges)
cv2.waitKey(0)
cv2.imshow('combined sobel edges', combined_sobel_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()