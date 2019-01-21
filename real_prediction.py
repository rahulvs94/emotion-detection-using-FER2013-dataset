import os
import cv2
import numpy as np
from emotion_detection import EmotionDetector

print(os.getcwd())
os.chdir('models')
print(os.getcwd())
model_name = 'tiny_XCEPTION'
a = EmotionDetector()
vidcap = cv2.VideoCapture(0)

while True:
    ret, image = vidcap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Predictions
    input_data = np.resize(image, [1, 48, 48, 3])
    pred_array = a.prediction(model_path=model_name+'.h5', img=input_data)[0]*100
    Emotions = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    print(
        "Angry: {0:.3f}%\nFear: {1:.3f}%\nHappy: {2:.3f}%\nSad: {3:.3f}%\nSurprise: {4:.3f}%\nNeutral: {5:.3f}%".format(
            pred_array[0],
            pred_array[1],
            pred_array[2],
            pred_array[3],
            pred_array[4],
            pred_array[5]
        ))

    # using openCV
    cv2.putText(image, str(Emotions[np.where(pred_array == max(pred_array))[0][0]]), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.imshow("Response", image)
    cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
vidcap.release()
cv2.destroyAllWindows()
