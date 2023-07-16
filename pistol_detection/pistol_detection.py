"""
Created by Berkay Taskin
YoloV8 Pistol-Detection
"""

from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)  # 0 yazılırsa laptop kamerasını başlatma veya yüklenilen videoyu eklerse videoyu başlatma
# örnek cap = cv2.VideoCapture('vtest4.mp4')

# Kameradan alınacak görüntünün genişlik ve yüksekliğini alma cap.get(3) genişlik cap.get(4) yükseklik
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Video dosyasını oluşturma ve ayarlarını belirleme
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

model = YOLO("best.pt")  # Kendi verisetiyle eğittiğim YOLO modelini yükleme
classNames = ["tabanca"]  # Sınıf ismini belirleme

while True:
    success, img = cap.read()  # Kameradan bir kare okuma

    # YOLOv8 kullanarak kareleri tespit etme
    # stream = True, daha verimli bir şekilde çalışmasını sağlar
    # conf=0.6, modelin tahmini gerçek bir tahmin olarak kabul edeceği minimum puandır ve
    # 0.6 değeri yüzde 60'ın üstündeki emin olunan tahminleri gösterir
    results = model(img, stream=True, conf=0.6)

    # Tespit sonuçlarını kontrol etme
    for r in results:
        # Her bir tespit sonucunda sınırlayıcı kutuları alırız
        boxes = r.boxes
        # Her bir sınırlayıcı kutu için işlemleri yapma
        for box in boxes:
            # tespit etiketinin koordinatlarını alıp dikdörtgen çizme
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Güven değerini alıp yazdırma
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name}{conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]

            # Etiketin boyutunu hesaplayıp arkaplanı çizme
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)

            # Etiketi yazma
            cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    # Görüntüyü video dosyasına yazma
    out.write(img)
    # Görüntüyü gösterme
    cv2.imshow("Image", img)

    # '1' tuşuna basıldığında döngüyü sonlandırma hangi tuş isterseniz yazabilirsiniz ---> ord('1')
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break
# Video yazma nesnesini serbest bırakma
out.release()
