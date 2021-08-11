import os
import PIL.Image as Image

def resizeImage(img, size):
    img_name = img.split("/")[-1]
    img = Image.open(img)
    img_res = img.resize(size)
    return img_res, img_name

def resizeVideoFolder(vid_folder, size):
    vid_folder_name = vid_folder.split("/")[-1]
    vid_res = []
    for img in sorted(os.listdir(vid_folder)):
        img_path = os.path.join(vid_folder, img)
        img_res, img_name = resizeImage(img_path, size)
        vid_res.append([img_res, img_name])
    return vid_res, vid_folder_name

def iterFolder(src_folder, size, clip):
    # clip => (start_id, end_id)
    new_folder = src_folder + f"_{size[0]}"
    vids = sorted(os.listdir(src_folder))
    i = clip[0]
    for vid in vids[clip[0]:clip[1]]:
        i += 1
        vid_folder = os.path.join(src_folder, vid)
        vid_res, vid_folder_name = resizeVideoFolder(vid_folder, size)
        for img_data in vid_res:
            img_res, img_name = img_data
            new_vid_folder = os.path.join(new_folder, vid_folder_name)
            if not os.path.exists(new_vid_folder):
                os.makedirs(new_vid_folder)
            new_img_name = os.path.join(new_vid_folder, img_name)
            saveImage(img_res, new_img_name)
        if i % 10 == 0:
            print(f"{i}/{len(vids)}", vid_folder_name, " is created")

def saveImage(img, img_name):
    img.save(img_name)

def resizeVideo(video, size, save):
    import imutils  # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
    import cv2  # opencv 모듈

    """
        :param video    : path  ~/.yourpath/video1.mp4
        :param size     : tuple (width, height)
        :param save     : path  ~/.savepath/video_out.mp4
    """

    # 비디오 파일
    video = video  # "" 일 경우 webcam 사용

    # 저장할 비디오 파일 경로 및 이름
    result_path = save

    # 비디오 경로가 제공되지 않은 경우 webcam 사용
    if video == "":
        print("[webcam 시작]")
        vs = cv2.VideoCapture(0)

    # 비디오 경로가 제공된 경우 video 사용
    else:
        print("[video 시작]")
        vs = cv2.VideoCapture(video)

    # 비디오 저장 변수
    writer = None

    # 비디오 스트림 프레임 반복
    while True:
        # 프레임 읽기
        ret, frame = vs.read()

        # 읽은 프레임이 없는 경우 종료
        if frame is None:
            break

        # 프레임 resize
        frame = cv2.resize(frame, size)

        # 프레임 출력
        cv2.imshow("frame", frame)

        # 'q' 키를 입력하면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # 저장할 비디오 설정
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"DIVX")
            writer = cv2.VideoWriter(result_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

        # 비디오 저장
        if writer is not None:
            writer.write(frame)

    # 종료
    vs.release()
    cv2.destroyAllWindows()


src_7 = "/home/butlely/Desktop/Dataset/aihub/source_7"
src_8 = "/home/butlely/Desktop/Dataset/aihub/source_8"
src_9 = "/home/butlely/Desktop/Dataset/aihub/source_9"
rowoon = "/home/butlely/PycharmProjects/mmlab/mmpose/demo/resources/rowoon.mp4"
rowoon2 = "/home/butlely/PycharmProjects/mmlab/mmpose/demo/resources/rowoon_256.mp4"
ksh = "/home/butlely/PycharmProjects/mmlab/mmpose/demo/resources/ksh_youtube3.mp4"
ksh_out = "/home/butlely/PycharmProjects/mmlab/mmpose/demo/resources/ksh_out.mp4"

# resizeVideo(video=ksh, size=(256,256), show=False, save=ksh_out)
iterFolder(src_9, (256,256), [4500,5996])