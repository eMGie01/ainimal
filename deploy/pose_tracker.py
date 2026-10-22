import os
import cv2
from ultralytics import YOLO


def detect_dog_pose(input_dir, output_dir):
  os.makedirs(output_dir, exist_ok=True)

  # Ładujemy lekki, wstępnie trenowany model YOLOv8n-pose
  print('Ładowanie modelu YOLOv8-Pose...')
  model = YOLO('yolov8n-pose.pt')

  # Przechodzimy przez wszystkie pliki w folderze z ostrymi klatkami
  for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_dir, filename)

        # Odczytujemy obraz przez OpenCV
        frame = cv2.imread(img_path)

        # Przepuszczamy obraz przez model YOLOv8-Pose
        results = model(frame)

        # Sprawdźmy, ile wykryto obiektów z kluczowymi punktami
        print(f"Lykryto obiektów: {len(results[0].boxes)}")

        # Rysujemy wykryty szkielet na obrazie
        annotated_frame = results[0].plot()

      # Zapisujemy wynik do folderu wyjściowego
        out_path = os.path.join(output_dir, f'pose_{filename}')
        cv2.imwrite(out_path, annotated_frame)
        print(f'Przetworzono i zapisano: {out_path}')


if __name__ == '__main__':
    detect_dog_pose(
        input_dir='C:/Users/dziur/Documents/GitHub/ainimal/data/extracted_frames/Siting',
        output_dir='C:/Users/dziur/Documents/GitHub/ainimal/data/extracted_frames/Output/Siting',
    )

    detect_dog_pose(
        input_dir='C:/Users/dziur/Documents/GitHub/ainimal/data/extracted_frames/Standing',
        output_dir='C:/Users/dziur/Documents/GitHub/ainimal/data/extracted_frames/Output/Standing',
    )