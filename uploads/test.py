import cv2
import numpy as np

# 全局变量
ref_point = []
cropping = False
image = None
clone = None

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # 画矩形
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Image", image)

def compare_images_in_roi(image1, image2, roi):
    x, y, w, h = roi
    roi1 = image1[y:y+h, x:x+w]
    roi2 = image2[y:y+h, x:x+w]
    cv2.imshow("roi1", roi1)
    cv2.imshow("roi2", roi2)

    # 调整大小确保一致
    roi1 = cv2.resize(roi1, (roi2.shape[1], roi2.shape[0]))

    # 转换为灰度图
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    # 计算差异
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 可视化差异
    colored_diff = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    result = np.hstack((roi1, roi2, colored_diff))
    cv2.imshow("Difference in ROI", result)

def main():
    global image, clone

    # 读取图片
    path1 = "D:\\Snipaste_img1.png"
    path2 = "D:\\Snipaste_img2.png"
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        print("图片路径错误，请重新输入。")
        return

    # 统一尺寸(376, 668)
    target_size = (min(img1.shape[1], img2.shape[1]), min(img1.shape[0], img2.shape[0]))
    print(target_size)
    img1 = cv2.resize(img1, target_size, interpolation=cv2.INTER_LINEAR)
    img2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_LINEAR)

    # 合并显示
    # combined = np.hstack((img1, img2))
    # cv2.imshow("Combined", combined)

    # 选择ROI区域
    image = img1.copy()
    clone = image.copy()
    # cv2.namedWindow("Image")
    # cv2.setMouseCallback("Image", click_and_crop)

    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF

        roi = (28, 253, 110, 44)
        compare_images_in_roi(img1, img2, roi)



if __name__ == "__main__":
    main()
