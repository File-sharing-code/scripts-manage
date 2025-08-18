#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import base64
import pickle
import json
import gzip
import cv2
import ast
import os


def load_images(path1, path2):
    """
    将两张图片进行灰度处理
    """
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1_new, img2_new = resize_images(img1, img2)
    return img1_new, img2_new


def load_config(config_path):
    """
    加载配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data


def init_load_images(root_dir, config_path):
    """
    初始化配置文件（更新图片配置、坐标）
    """
    pic_list = []
    pic_info = {}
    design_info = {}
    for fpath, dirs, fs in os.walk(root_dir):
        for f in fs:
            pic_list.append(os.path.join(fpath, f))

    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for pic in pic_list:
        pic_tail = os.path.split(pic)
        if os.path.split(pic_tail[-2])[-1] in pic_info:
            pic_info[os.path.split(pic_tail[-2])[-1]]["pics"].append(pic)
        else:
            pic_info[os.path.split(pic_tail[-2])[-1]] = {"pics": [pic]}
        if "rois" not in pic_info[os.path.split(pic_tail[-2])[-1]]:
            pic_info[os.path.split(pic_tail[-2])[-1]]["rois"] = []
        if "ignores" not in pic_info[os.path.split(pic_tail[-2])[-1]]:
            pic_info[os.path.split(pic_tail[-2])[-1]]["ignores"] = []

    for key, val in pic_info.items():
        roi_dict = {}
        if "rois" in data["info"][key]:
            rois = data["info"][key]["rois"]
            pics = val["pics"]
            val["rois"] = rois
            for roi in rois:
                roi_list = []
                for pic in pics:
                    ior_img = pick_roi_image(pic, roi)
                    unpick_roi_image(ior_img)
                    roi_list.append(ior_img)
                roi_dict[roi] = roi_list
            design_info[key] = roi_dict
        if "ignores" in data["info"][key]:
            ignores = data["info"][key]["ignores"]
            val["ignores"] = ignores

    data["info"].update(pic_info)
    data["design"].update(design_info)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def pick_roi_image(ipath, roi):
    """
    压缩截取指定区域图片
    """
    image = cv2.imread(ipath)
    x, y, w, h = ast.literal_eval(roi)
    compressed_data = gzip.compress(pickle.dumps(image[y:y+h, x:x+w]))
    return base64.b64encode(compressed_data).decode('utf-8')


def unpick_roi_image(compressed_data):
    """
    解压缩截取指定区域图片
    """
    compressed_data = base64.b64decode(compressed_data)
    uncompressed_data = pickle.loads(gzip.decompress(compressed_data))
    return uncompressed_data


def resize_images(img1, img2):
    """
    将两张图片缩放为相同大小（以较小图片的尺寸为准）
    """
    min_width = min(img1.shape[1], img2.shape[1])
    min_height = min(img1.shape[0], img2.shape[0])

    img1_resized = cv2.resize(img1, (min_width, min_height), interpolation=cv2.INTER_LINEAR)
    img2_resized = cv2.resize(img2, (min_width, min_height), interpolation=cv2.INTER_LINEAR)

    return img1_resized, img2_resized


def compare_images_with_config(config_data: dict, module, image, thresh=127, debug=False):
    act_result = True
    all_result_info = {}
    design_result_info = {}
    pics = config_data["info"][module]["pics"]          # 样本图片
    rois = config_data["info"][module]["rois"]          # 指定区域
    ignores = config_data["info"][module]["ignores"]    # 忽略区域
    design_rois = config_data["design"][module]         # 样本区域
    ignore_list = [ast.literal_eval(area) for area in ignores + rois]
    img1, img2 = load_images(pics[0], image)
    result = compare_images_with_ignore(img1, img2, ignore_list, thresh, debug)
    if result is None:
        all_result_info["ignore"] = True
    else:
        all_result_info["ignore"] = False

    act_result = act_result and all_result_info["ignore"]

    # 对比相似度
    for roi in rois:
        design_res = False
        x, y, w, h = ast.literal_eval(roi)
        roi1 = img2[y:y+h, x:x+w]
        for design in design_rois[roi]:
            # 解压缩成image array
            unpick_design = unpick_roi_image(design)
            result = compare_images_similar_ssim(roi, roi1, unpick_design)
            if result is None:
                design_res = True
                break
        design_result_info[roi] = design_res
        act_result = act_result and design_res
    all_result_info["design"] = design_result_info
    all_result_info["ignore_area"] = ignores
    all_result_info["image_info"] = [pics[0], image]

    return act_result, all_result_info


def compare_images_with_ignore(image1, image2, ignore, thresh=127, debug=False, show=False):
    """
    对比两张图片差异性（可设置忽略对比区域）, show=True时专门提供给show_compare_images_result方法使用
    """
    # 转换为灰度图
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算差异
    diff = cv2.absdiff(img1_gray, img2_gray)

    # 将忽略区域设为0（不显示差异）
    mask = np.zeros_like(diff)
    for (x, y, w, h) in ignore:
        mask[y:y+h, x:x+w] = 255
    diff_masked = cv2.bitwise_and(diff, diff, mask=~mask)

    # 计算差异
    _, thresh = cv2.threshold(diff_masked, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if show:
        return contours

    if debug:
        result = image1.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("Difference with Ignore", result)
        cv2.waitKey(0)

    if len(contours) > 0:
        return False


def compare_images_in_roi(image1, image2, roi, thresh=127, debug=False):
    """
    对比两张图片差异性（可设置只对比设定区域）
    """
    x, y, w, h = roi
    roi1 = image1[y:y+h, x:x+w]
    roi2 = image2[y:y+h, x:x+w]

    # 比较相似度
    is_similar = compare_images_similar_ssim(roi, roi1, roi2)
    print(is_similar)
    is_similar = compare_images_similar_norm(roi, roi1, roi2)
    print(is_similar)

    # 转换为灰度图
    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    # 计算差异
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    # 可视化差异
    colored_diff = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    result = np.hstack((roi1, roi2, colored_diff))

    if debug:
        cv2.imshow("Difference in ROI", result)
        cv2.waitKey(0)


def compare_images_similar_ssim(region, roi1, roi2, threshold=0.9):
    """
    使用结构相似性（SSIM）作为对比
    """
    from skimage.metrics import structural_similarity as ssim
    # 调整大小确保一致
    roi1 = cv2.resize(roi1, (roi2.shape[1], roi2.shape[0]))

    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    score, diff = ssim(gray1, gray2, full=True)
    print(f"[{str(region)}区域相似度] SSIM: {score:.4f}")

    if score < threshold:
        return False


def compare_images_similar_norm(region, roi1, roi2, threshold=50):
    """
    使用矩阵或向量的范数（NORM）作为对比
    """
    # 调整大小确保一致
    roi1 = cv2.resize(roi1, (roi2.shape[1], roi2.shape[0]))

    gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    diff_value = cv2.norm(gray1, gray2, cv2.NORM_L2)
    print(f"选定区域{str(region)}差异值（L2范数）: {diff_value:.2f}")

    if diff_value > threshold:
        return False


def show_compare_images_result(info: dict, thresh=127):
    # 加载图片
    img1, img2 = load_images(info["image_info"][0], info["image_info"][1])
    # 画忽略区域矩形
    copy_img1 = img1.copy()
    copy_img2 = img2.copy()
    for ignore in info["ignore_area"]:
        x, y, w, h = ast.literal_eval(ignore)
        ref_point_1 = (x, y)
        ref_point_2 = (x + w, y + h)
        cv2.rectangle(copy_img1, ref_point_1, ref_point_2, (0, 255, 0), 2, 4)
        cv2.rectangle(copy_img2, ref_point_1, ref_point_2, (0, 255, 0), 2, 4)

    if not info["ignore"]:
        ignore_list = [ast.literal_eval(area) for area in info["ignore_area"]]
        contours = compare_images_with_ignore(copy_img1, copy_img2, ignore_list, thresh, show=True)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(copy_img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 画指定区域矩形
    for design, val in info["design"].items():
        x, y, w, h = ast.literal_eval(design)
        ref_point_1 = (x, y)
        ref_point_2 = (x + w, y + h)
        cv2.rectangle(copy_img1, ref_point_1, ref_point_2, (255, 0, 0), 2, 4)
        cv2.rectangle(copy_img2, ref_point_1, ref_point_2, (255, 0, 0), 2, 4)
        if not val:
            cv2.putText(copy_img2, " NoPass", ref_point_2, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    combined = np.hstack((copy_img1, copy_img2))
    cv2.imshow("Image", combined)
    cv2.waitKey(0)


if __name__ == '__main__':
    # 图片路径
    img1_path = "D:\\Snipaste_img1.png"
    img2_path = "D:\\Snipaste_img2.png"
    img3_path = "D:\\Snipaste_img3.png"
    config_path = ".\\image_config.json"

    init_load_images(".\\imgs", config_path)
    #
    config = load_config(config_path)
    res, res_info = compare_images_with_config(config, "QuoteChart", img3_path, debug=False)
    print(res, res_info)

    # img_1, img_2 = load_images(img1_path, img2_path)

    # 忽略区域列表 [(x, y, w, h), (18, 195, 330, 105), ...]
    # ignore_regions = [(18, 195, 330, 105)]
    # res = compare_images_with_ignore(img_1, img_2, ignore_regions, debug=False)
    # print(res)

    # 对比特定区域 [(x, y, w, h), (30, 257, 104, 23), (80, 392, 55, 56)...]
    # roi_regions = [(80, 392, 55, 56)]
    # compare_images_in_roi(img_1, img_2, roi_regions[0])

    # 执行配置文件
    show_compare_images_result(res_info)
