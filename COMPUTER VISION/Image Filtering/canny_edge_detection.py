import cv2
import numpy as np

image = cv2.imread('/home/sathvik/PycharmProjects/opencv/Images/lena.jpeg', 0)
cv2.imshow('original image', image)
width, height = image.shape
new_width = int(width*2.5)
new_height = int(height*2.5)
image = cv2.resize(image, (new_height, new_width))
cv2.waitKey(0)

gaussian_image = cv2.GaussianBlur(image, (5, 5), 1)
cv2.imshow('blurred image', gaussian_image)
cv2.waitKey(0)

# Compute gradient of the images
horizontal_edges = cv2.Sobel(gaussian_image, cv2.CV_64F, 1, 0, ksize=3)
vertical_edges = cv2.Sobel(gaussian_image, cv2.CV_64F, 0, 1, ksize=3)


cv2.imshow('horizontal edges', horizontal_edges)
cv2.waitKey(0)
cv2.imshow('vertical edges', vertical_edges)
cv2.waitKey(0)

# Compute magnitude and direction
Magnitude = np.sqrt(horizontal_edges**2 + vertical_edges**2)
direction = np.arctan2(vertical_edges, horizontal_edges) * (180.0 / np.pi)  # Convert to degrees

cv2.imshow('gradient Magnitude before NMS', (Magnitude / Magnitude.max() * 255).astype(np.uint8))
cv2.waitKey(0)


def non_maximum_suppression(gradient_magnitude, gradient_direction):
    M, N = gradient_magnitude.shape
    output = np.zeros((M, N), dtype=np.float32)

    # Convert gradient direction to [0, 180] range
    gradient_direction = gradient_direction % 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            direction = gradient_direction[i, j]

            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                neighbors = [gradient_magnitude[i, j+1], gradient_magnitude[i, j-1]]
            elif 22.5 <= direction < 67.5:
                neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
            elif 67.5 <= direction < 112.5:
                neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
            elif 112.5 <= direction < 157.5:
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]

            if gradient_magnitude[i, j] >= max(neighbors):
                output[i, j] = gradient_magnitude[i, j]

    return output

def hysteresis_thresholding(gradient_magnitude, high_threshold, low_threshold):
    M, N = gradient_magnitude.shape
    # Initialize output image
    edges = np.zeros((M, N), dtype=np.uint8)

    # Step 1: Mark strong edges
    strong_edges = (gradient_magnitude > high_threshold)

    # Step 2: Mark weak edges
    weak_edges = (gradient_magnitude > low_threshold) & (gradient_magnitude <= high_threshold)

    # Step 3: Edge tracking by hysteresis
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if strong_edges[i, j]:
                edges[i, j] = 255  # Strong edge
            elif weak_edges[i, j]:
                # Check if any of the 8-connected neighbors are strong edges
                if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                    edges[i, j] = 255  # Weak edge connected to a strong edge
                else:
                    edges[i, j] = 0  # Weak edge not connected to strong edge

    return edges


suppressed_image = non_maximum_suppression(Magnitude, direction)
cv2.imshow('suppressed image', suppressed_image / suppressed_image.max())  # Normalize for display
cv2.waitKey(0)

final_edges = hysteresis_thresholding(suppressed_image, 50, 30)
cv2.imshow('final edges', final_edges)
cv2.waitKey(0)

cv2.destroyAllWindows()
