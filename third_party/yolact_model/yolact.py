import sys
import os
_FILEPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_FILEPATH)

import argparse
import random

import torch
import torch.onnx
import torch.backends.cudnn as cudnn

import onnxruntime as rt
import cv2

from data import COLORS
from data import cfg, set_cfg
from utils.augmentations import FastBaseTransform
from utils import timer
from layers import Detect
from layers.output_utils import postprocess, undo_image_transformation

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')

    global args

    args = parser.parse_args(argv)

    setArgs()

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed)

def setArgs():
    def prefix(str):
        return os.path.join(_FILEPATH, str)
    args.trained_model = prefix("weights/yolact_base_54_800000.pth")
    args.top_k = 100
    args.cuda = False
    args.fast_nms = True
    args.display = False
    args.shuffle = False
    args.ap_data_file = prefix("results/ap_data.pkl")
    args.resume = False
    args.max_images = -1
    args.output_coco_json = False
    args.bbox_det_file = prefix("results/bbox_detections.json")
    args.mask_det_file = prefix("results/mask_detections.json")
    args.config = "yolact_base_config"
    args.output_web_json = False
    args.web_det_path = "web/dets/"
    args.no_bar = False
    args.display_lincomb = False
    args.benchmark = False
    args.no_sort = False
    args.seed = None
    args.images = None
    args.video = None
    args.video_multiframe = 1
    args.score_threshold = 0.3
    args.dataset = None
    args.detect = False
    args.no_hash = False
    args.mask_proto_debug = False
    args.crop = True
    # display settings
    args.display_masks = True
    args.display_bboxes = True
    args.display_text = False
    args.display_scores = False



def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45):
    """
    todo: return bbox, maskimg_numpy
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy)
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape

    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                        crop_masks=args.crop,
                        score_threshold=args.score_threshold)
        # torch.cuda.synchronize()

    with timer.env('Copy'):
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][:args.top_k]
        classes, scores, boxes = [x[:args.top_k].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return ((img_gpu * 255).byte().cpu().numpy(), [], [])

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)

        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if args.display_masks and cfg.eval_mask_branch:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]

        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat(
            [(torch.Tensor(get_color(j)).float() / 255).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3).cuda() * colors.cuda() * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1

        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider - 1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul.cuda()
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu.cuda() * inv_alph_masks.prod(dim=0).cuda() + masks_color_summand.cuda()

    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    if args.display_text or args.display_bboxes:
        for j in reversed(range(num_dets_to_consider)):
            x1, y1, x2, y2 = boxes[j, :]
            color = get_color(j)
            score = scores[j]

            if args.display_bboxes:
                cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

            if args.display_text:
                _class = cfg.dataset.class_names[classes[j]]
                text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                font_thickness = 1

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

                text_pt = (x1, y1 - 3)
                text_color = [255, 255, 255]

                cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
                cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness,
                            cv2.LINE_AA)
    if len(t[2]) > 0:
        bbox = t[2][0]
        mask = t[3][0]
    else:
        bbox = []
        mask = []
    return (img_numpy, bbox, mask)


def evalimage(img, sess = None, sess_info = None):
    # sess_info => [[loc_name, conf_name, mask_name, priors_name, proto_name], input_name]
    frame = torch.from_numpy(img).cuda().float()
    transNet = FastBaseTransform()
    transNet.cuda()
    batch = transNet(frame.unsqueeze(0))

    pred_onnx = sess.run(sess_info[0],
                        {sess_info[1]: batch.cpu().numpy()})

    # priors = np.loadtxt('priors.txt', delimiter=',', dtype='float32')

    detect = Detect(cfg.num_classes, bkg_label=0, top_k=200, conf_thresh=0.05, nms_thresh=0.5)
    preds = detect({'loc': torch.from_numpy(pred_onnx[0]), 'conf': torch.from_numpy(pred_onnx[1]),
                    'mask': torch.from_numpy(pred_onnx[2]), 'priors': torch.from_numpy(pred_onnx[3]),
                    'proto': torch.from_numpy(pred_onnx[4])})

    (img_numpy, bbox, mask) = prep_display(preds, frame, None, None, undo_transform=False)
    return (img_numpy, bbox, mask)


def setYolactOnnx(onnx):
    sess = rt.InferenceSession(onnx)

    input_name = sess.get_inputs()[0].name
    loc_name = sess.get_outputs()[0].name
    conf_name = sess.get_outputs()[1].name
    mask_name = sess.get_outputs()[2].name
    priors_name = sess.get_outputs()[3].name
    proto_name = sess.get_outputs()[4].name
    sess_info = [[loc_name, conf_name, mask_name, priors_name, proto_name], input_name]
    return sess, sess_info

def setYolact(onnx):
    parse_args()

    set_cfg(args.config)
    cfg.mask_proto_debug = args.mask_proto_debug

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

    sess, sess_info = setYolactOnnx(onnx)
    return sess, sess_info

def runYolact(img, sess, sess_info):
    (img_np, bbox, mask) = evalimage(img, sess=sess, sess_info=sess_info)
    return (img_np, bbox, mask)


