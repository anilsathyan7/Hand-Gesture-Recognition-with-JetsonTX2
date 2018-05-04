import numpy as np
import cv2

cap = cv2.VideoCapture(0)
i=1
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (7,7), 3)
    frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(frame, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Display the resulting frame
    cv2.imshow('frame',new)
    img_name='zeru' + str(i) + '.png'
    if i<6000 :
      cv2.imwrite(img_name, new)
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
