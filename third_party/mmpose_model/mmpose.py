import onnx
import onnxruntime
import numpy as np
from .transforms import compose, resizeData, flip_back, reformCoord
from .config import img_metas
from .utils import decode
from .show import putCircle

def setMmpose(onnx_file):
    model = onnx.load(onnx_file)
    input_all = [node.name for node in model.graph.input]
    input_initializer = [
        node.name for node in model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    ort_session = onnxruntime.InferenceSession(onnx_file)
    return (ort_session, net_feed_input)

def runModel(img, img_flipped, sess, sess_info, img_metas):
    # run model
    heatmap = sess.run(None, {sess_info[0]: img})
    heatmap = np.round(heatmap[0], 4)
    # okay!!
    heatmap_flipped = sess.run(None, {sess_info[0]: img_flipped})
    heatmap_flipped = heatmap_flipped[0]
    heatmap_flipped = flip_back(heatmap_flipped)
    output_heatmap = (heatmap + heatmap_flipped) * 0.5

    predict = decode(img_metas, output_heatmap)
    coords = predict['preds'][0]
    return coords

def runMmpose(img, bbox, sess, sess_info):
    hasBbox = ~isinstance(bbox, list)
    coords = []
    if not hasBbox:
        return img, coords
    try:
        bbox[0]
    except:
        return img, coords
    resized_img = resizeData(img, bbox)
    trans_img, fliped_img = compose(resized_img, img_metas)
    coords = runModel(trans_img, fliped_img, sess, sess_info, img_metas)
    coords = reformCoord(coords, bbox)
    img = putCircle(img, coords)
    return img, coords
