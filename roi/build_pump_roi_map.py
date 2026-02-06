import json
from pathlib import Path

COCO_JSON = Path("annotations.json")   # CVAT export
OUTPUT_JSON = Path("pump_rois.json")

PUMP_CATEGORY_ID = 1  # your pump class

def build_roi_map(coco_path: Path):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # image_id → filename
    image_id_to_name = {
        img["id"]: img["file_name"]
        for img in coco["images"]
    }

    roi_map = {}

    for ann in coco["annotations"]:
        if ann["category_id"] != PUMP_CATEGORY_ID:
            continue

        image_id = ann["image_id"]
        filename = image_id_to_name[image_id]

        x, y, w, h = ann["bbox"]

        # Round & cast once here
        roi_map[filename] = [
            int(round(x)),
            int(round(y)),
            int(round(w)),
            int(round(h)),
        ]

    return roi_map


if __name__ == "__main__":
    roi_map = build_roi_map(COCO_JSON)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(roi_map, f, indent=2)

    print(f"Saved {len(roi_map)} pump ROIs to {OUTPUT_JSON}")
