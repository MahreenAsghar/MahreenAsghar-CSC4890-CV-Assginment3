import cv2
import matplotlib.pyplot as plt

correlation_img = cv2.imread("capture_isp_4.png")
snip_image = correlation_img[714:2228,341:1420]

test_images = [
    "capture_isp_1.png",
    "capture_isp_5.png",
    "capture_isp_3.png",
    "capture_isp_2.png",
    "capture_isp_10.png",
    "capture_isp_9.png",
    "capture_isp_6.png",
    "capture_isp_8.png",
    "capture_isp_4.png",
    "capture_isp_7.png"
]

correlations = []

for img in test_images:
    testImg = cv2.imread(img)
    croppedTestImg = testImg[714:2228,341:1420]
    plt.imshow(croppedTestImg)
    plt.show()
    X = croppedTestImg - snip_image
    ssd = sum(X[:]**2)
    correlations.append(ssd)

print(correlations)


cv2.imshow("Correlation Img",correlation_img)
plt.imshow (snip_image)
plt.show()