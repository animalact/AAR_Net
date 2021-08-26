import numpy as np

img_metas = [
    {
        "image_file": '',
        "center": np.array([128.5, 128.5]),
        "scale": np.array([1.6062499, 1.6062499]),
        "rotation": 0,
        "bbox_score": 1,
        'flip_pairs': [[5, 6], [7, 8], [9, 10], [11, 12]],
        "bbox_id": 0,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "size": (256, 256)
     }
]