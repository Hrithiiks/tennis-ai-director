




import cv2
import matplotlib.pyplot as plt

cap1 = cv2.VideoCapture(r"C:\Users\HRITHIK S\TENNIS AI DIRECTOR\Recorded Videos\baseline\BL_set1.MOV")
cap2 = cv2.VideoCapture(r"C:\Users\HRITHIK S\TENNIS AI DIRECTOR\Recorded Videos\sideline\SL_set1.mp4")
cap3 = cv2.VideoCapture(r"C:\Users\HRITHIK S\TENNIS AI DIRECTOR\Recorded Videos\corner top view\TC_set1.MOV")
# Show first 10 frames for sync check
for i in range(10):
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()
    ret3, f3 = cap3.read()

    if not (ret1 and ret2 and ret3):
        break

    # Resize all frames to same size
    f1 = cv2.resize(f1, (426, 240))
    f2 = cv2.resize(f2, (426, 240))
    f3 = cv2.resize(f3, (426, 240))

    # Combine horizontally
    row = cv2.hconcat([f1, f2, f3])
    row_rgb = cv2.cvtColor(row, cv2.COLOR_BGR2RGB)

    # Show with matplotlib
    plt.imshow(row_rgb)
    plt.title(f"Frame {i}")
    plt.axis("off")
    plt.show()

cap1.release()
cap2.release()
cap3.release()

