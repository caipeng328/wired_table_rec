import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from pathlib import Path
from TABLE_REC import WiredTableRecognition
from rapidocr import RapidOCR
import cv2
import os
import time
import numpy as np
import json


def draw_polygons(image, polygons, color=(0, 255, 0), thickness=2):
    for poly in polygons:
        poly = np.array(poly, dtype=np.int32)
        pts = poly.reshape(-1, 1, 2)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
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

def process_folder(
    img_folder: str, 
    save_html_folder: str, 
    save_vis_folder: str,
    save_json_folder: str
):
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
                ocr_result = list(
                    zip(rapid_ocr_output.boxes, rapid_ocr_output.txts, rapid_ocr_output.scores)
                )
            except:
                ocr_result = None

            t = time.time()
            table_results = wired_engine(str(img_path), ocr_result=ocr_result)
            print(f"Inference time: {time.time() - t:.2f}s")

            html_content = table_results[0]
            sorted_polygons = table_results[2]
            sorted_logi_points = table_results[3]

            if html_content != "":
                html_name = img_path.stem + ".html"
                with open(save_html_folder / html_name, "w", encoding="utf-8") as f:
                    f.write(html_content)

            img = cv2.imread(str(img_path))
            vis_img = draw_polygons(img, sorted_polygons)
            vis_name = img_path.stem + "_vis.jpg"
            cv2.imwrite(str(save_vis_folder / vis_name), vis_img)

            json_name = img_path.stem + ".json"
            save_json_file(save_json_folder / json_name, sorted_polygons, sorted_logi_points)
        except:
            continue

if __name__ == "__main__":
    input_folder = "test_image"                
    output_html_folder = "output_htmls"        
    output_vis_folder = "output_visuals"      
    output_json_folder = "output_jsons" 

    process_folder(input_folder, output_html_folder, output_vis_folder, output_json_folder)
