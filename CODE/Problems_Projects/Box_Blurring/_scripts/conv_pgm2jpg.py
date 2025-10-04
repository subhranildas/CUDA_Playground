import cv2

# Load PGM image (grayscale)
img = cv2.imread("blurred.pgm", cv2.IMREAD_GRAYSCALE)

# Save as PNG
cv2.imwrite("output.png", img)

