import onnx
import onnxruntime
import cv2
import numpy as np

from third_party.mmpose_model.config import img_metas
from third_party.mmpose_model.utils import _get_max_preds, transform_preds, keypoints_from_heatmaps, decode
from third_party.mmpose_model.transforms import get_affine_transform, trans_affine, trans_reshape, trans_normalize, trans_expand, flip_back
from mmpose_model.show import putCircle



def main(video, save="", show=False):
    onnx_file = '/home/butlely/PycharmProjects/AAR_Net/weights/mmpose/onnx/poodle_w32.onnx'
    model = onnx.load(onnx_file)
    check = onnx.checker.check_model(model)
    graph = onnx.helper.printable_graph(model.graph)
    # weights = model.graph.initializer
    input_all = [node.name for node in model.graph.input]
    input_initializer = [
        node.name for node in model.graph.initializer
    ]
    net_feed_input = list(set(input_all) - set(input_initializer))
    ort_session = onnxruntime.InferenceSession(onnx_file)

    input_name = ort_session.get_inputs()[0].name

    result_path = save

    if video == "":
        print("[webcam 시작]")
        vs = cv2.VideoCapture(0)
    else:
        print("[video 시작]")
        vs = cv2.VideoCapture(video)

    writer = None

    while True:
        ret, frame = vs.read()
        if frame is None:
            break

        # my code
        img = frame.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # transform
        img = trans_affine(img, img_metas[0]['center'], img_metas[0]['scale'], img_metas[0]['rotation'], img_metas[0]['size'])
        img = trans_reshape(img)
        img = trans_normalize(img, mean=img_metas[0]['mean'], std=img_metas[0]['std'])
        img = trans_expand(img)
        img = img.astype(np.float32)

        img_flipped = np.flip(img, 3)
        # run model
        heatmap = ort_session.run(None, {net_feed_input[0]: img})
        heatmap = np.round(heatmap[0],4)
        # okay!!
        heatmap_flipped = ort_session.run(None, {net_feed_input[0]: img_flipped})
        heatmap_flipped = heatmap_flipped[0]
        heatmap_flipped = flip_back(heatmap_flipped)
        output_heatmap = (heatmap + heatmap_flipped) * 0.5

        predict = decode(img_metas, output_heatmap)
        coords = predict['preds'][0]
        print(coords)
        frame = putCircle(frame, coords)

        # break
        # 프레임 출[0,0,20:25,20:25]력
        if show:
            cv2.imshow("frame", frame)

            # 'q' 키를 입력하면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        if save:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(result_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

            # 비디오 저장
            if writer is not None:
                writer.write(frame)

    # 종료
    vs.release()
    cv2.destroyAllWindows()

vid_path = "/home/butlely/PycharmProjects/mmlab/mmpose/works_dirs/002.mp4"
main(vid_path, show=True)
