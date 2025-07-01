import os
import sys
import argparse
import time
import json
from pathlib import Path

import cv2
import numpy as np

# 设置项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from TABLE_REC import WiredTableRecognition
from rapidocr import RapidOCR


def draw_polygons(image, polygons, color=(0, 255, 0), thickness=2):
    for poly in polygons:
        poly = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [poly], isClosed=True, color=color, thickness=thickness)
    return image


def save_json_file(json_path, sorted_polygons, sorted_logi_points):
    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [to_list(i) for i in obj]
        else:
            return obj

    json_data = {
        "polygons": to_list(sorted_polygons),
        "logi_points": to_list(sorted_logi_points)
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


def process_folder(img_folder, save_html_folder, save_vis_folder, save_json_folder):
    wired_engine = WiredTableRecognition()
    ocr_engine = RapidOCR()

    img_folder = Path(img_folder)
    save_html_folder = Path(save_html_folder)
    save_vis_folder = Path(save_vis_folder)
    save_json_folder = Path(save_json_folder)

    save_html_folder.mkdir(parents=True, exist_ok=True)
    save_vis_folder.mkdir(parents=True, exist_ok=True)
    save_json_folder.mkdir(parents=True, exist_ok=True)

    img_paths = list(img_folder.glob("*.png")) + list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.jpeg"))

    for img_path in img_paths:
        try:
            print(f"Processing: {img_path}")
            try:
                rapid_ocr_output = ocr_engine(str(img_path), return_word_box=True)
                ocr_result = list(zip(rapid_ocr_output.boxes, rapid_ocr_output.txts, rapid_ocr_output.scores))
            except Exception as e:
                print(f"OCR failed: {e}")
                ocr_result = None

            t = time.time()
            table_results = wired_engine(str(img_path), ocr_result=ocr_result)
            print(f"Inference time: {time.time() - t:.2f}s")

            html_content = table_results[0]
            sorted_polygons = table_results[2]
            sorted_logi_points = table_results[3]

            if html_content:
                with open(save_html_folder / f"{img_path.stem}.html", "w", encoding="utf-8") as f:
                    f.write(html_content)

            img = cv2.imread(str(img_path))
            vis_img = draw_polygons(img, sorted_polygons)
            cv2.imwrite(str(save_vis_folder / f"{img_path.stem}.jpg"), vis_img)

            save_json_file(save_json_folder / f"{img_path.stem}.json", sorted_polygons, sorted_logi_points)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch wired table recognition.")
    parser.add_argument("--input_folder", type=str, default="test_image", help="Folder containing input images")
    parser.add_argument("--output_html_folder", type=str, default="output_htmls", help="Folder to save HTML outputs")
    parser.add_argument("--output_vis_folder", type=str, default="output_visuals", help="Folder to save visual outputs")
    parser.add_argument("--output_json_folder", type=str, default="output_jsons", help="Folder to save JSON outputs")
    args = parser.parse_args()

    process_folder(
        args.input_folder,
        args.output_html_folder,
        args.output_vis_folder,
        args.output_json_folder,
    )
