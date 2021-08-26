import onnx
import onnxruntime
import cv2
import numpy as np
import matplotlib.pyplot as plt

def getMmpose():
    onnx_file = '/home/butlely/PycharmProjects/mmlab/mmpose/works_dirs/poodle/mm_poodle.onnx'
    model = onnx.load(onnx_file)
    input_all = [node.name for node in model.graph.input]
    input_initializer = [
        node.name for node in model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    ort_session = onnxruntime.InferenceSession(onnx_file)
    return (ort_session, net_feed_input)

def transformMmpose(img, mmpose):
    from .butlely_utils import _get_max_preds, transform_preds, keypoints_from_heatmaps, decode
    from .butlely_transforms import get_affine_transform, trans_affine, trans_reshape, trans_normalize, trans_expand, \
        flip_back
    from .butlely_show import putCircle
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

    # transform
    img = trans_affine(img, img_metas[0]['center'], img_metas[0]['scale'], img_metas[0]['rotation'],
                       img_metas[0]['size'])
    img = trans_reshape(img)
    img = trans_normalize(img, mean=img_metas[0]['mean'], std=img_metas[0]['std'])
    img = trans_expand(img)
    img = img.astype(np.float32)

    ort_session, net_feed_input = mmpose

    img_flipped = np.flip(img, 3)
    # run model
    heatmap = ort_session.run(None, {net_feed_input[0]: img})
    heatmap = np.round(heatmap[0], 4)
    # okay!!
    heatmap_flipped = ort_session.run(None, {net_feed_input[0]: img_flipped})
    heatmap_flipped = heatmap_flipped[0]
    heatmap_flipped = flip_back(heatmap_flipped)
    output_heatmap = (heatmap + heatmap_flipped) * 0.5

    predict = decode(img_metas, output_heatmap)
    coords = predict['preds'][0]
    return coords


def reformCoord(coords, bbox):
    print(coords)
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2]) - x
    h = int(bbox[3]) - y

    assert w > 0 and h > 0

    fx = w/h
    fy = h/w

    if h > w:
        w_new = int(256 * fx)
        pad = int((256-w_new)/2)
        coords[:,0] -= pad
        coords = np.multiply(coords, [h/256, h/256, 1])
        coords = np.add(coords, [x, y, 0])

    if w > h:
        h_new = int(256 * fy)
        pad = int((256-h_new)/2)
        coords[:, 1] -= pad
        coords = np.multiply(coords, [w / 256, w / 256, 1])
        coords = np.add(coords, [x, y, 0])

    return coords


def resizeData(img, bbox):
    # img (h, w, c)
    """
    ['image_file', 'center', 'scale', 'bbox', 'rotation', 'joints_3d', 'joints_3d_visible', 'dataset', 'bbox_score', 'bbox_id', 'ann_info', 'img'])
    """
    x = int(bbox[0])
    y = int(bbox[1])
    x1 = int(bbox[2])
    y1 = int(bbox[3])
    w = x1-x
    h = y1-y
    assert w>0 and h>0

    img_clipped = img[y:y + h, x:x + w]
    try:
        if h > w:
            fx = w / h
            w_new = int(256 * fx)
            pad = int((256 - w_new) / 2)
            img_resize = cv2.resize(img_clipped, dsize=(w_new, 256))
            img_pad = np.pad(img_resize, ((0, 0), (pad, 256 - w_new - pad), (0, 0)), 'constant', constant_values=0)

        else:
            fy = h / w
            h_new = int(256 * fy)
            pad = int((256 - h_new) / 2)
            img_resize = cv2.resize(img_clipped, dsize=(256, h_new))
            img_pad = np.pad(img_resize, ((pad, 256 - h_new - pad), (0, 0), (0, 0)), 'constant', constant_values=0)

        return img_pad
    except:
        return None


def show(img, keypoints):
    plt.imshow(img)
    xs, ys = alignKeypoint(keypoints)
    plt.scatter(xs, ys)
    plt.show(block=False)
    plt.pause(0.01)
    plt.clf()


def alignKeypoint(keypoints):
    xs = []
    ys = []
    for keypoint in keypoints:
        xs.append(keypoint[0])
        ys.append(keypoint[1])
    return xs, ys