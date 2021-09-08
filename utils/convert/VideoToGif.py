import imageio
import cv2


vid = "/home/butlely/PycharmProjects/AAR_Net/output/tester/aar_tester.mp4"

cap = cv2.VideoCapture(vid)
image_lst = []

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_lst.append(frame_rgb)

    cv2.imshow('a', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert to gif using the imageio.mimsave method
imageio.mimsave('/home/butlely/PycharmProjects/AAR_Net/output/result.gif', image_lst[36:-30], fps=5)
print("Done")