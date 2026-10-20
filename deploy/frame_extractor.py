import os
import cv2

#Function that will extract the best frames from a given video to wanted output folder
def extract_best_frames(video_path, output_dir, blur_threshold=100.0) -> None: 
    
    #otwieranie pliku, pamietać dodać wyświetlanie błędu jak nie uda się otworzyć (edge cases?)
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    Frames = 0
    Saved_Frames = 0

    #pobieranie kolejnych klatek filmu (jedna za drugą w pętli do stprawdzenia klatka po klatce), warto pamiętać żeby przetwarzać co którąś klatkę, bo inaczej będzie to bardzo wolne i niepotrzebne, np. co 5 klatek

    while True:
        ret, frame = video.read() #ret boolean that shows if the frame was read successfully, frame - the actual frame

        if not ret:
            break

    #kolor w szarości dla łatwiejszego przetwarzania
        if Frames % 5 == 0:  # Process every 5th frame
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #test ostrości, sprawdzenie z wybranym progiem

            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            if laplacian_var >= blur_threshold:
                filename = os.path.join(output_dir, f"frame_{Saved_Frames:04d}.jpg")
                cv2.imwrite(filename, frame)
                Saved_Frames += 1


        Frames += 1
    #zamykanie pliku, zwolnienie pamieci komputera
    video.release()


if __name__ =="__main__":
    extract_best_frames('C:/Users/dziur/Documents/GitHub/ainimal/data/raw_videos/Standing.mp4',
                       'C:/Users/dziur/Documents/GitHub/ainimal/data/extracted_frames/Standing',
                       blur_threshold=100.0)

    extract_best_frames('C:/Users/dziur/Documents/GitHub/ainimal/data/raw_videos/Siting.mp4',
                        'C:/Users/dziur/Documents/GitHub/ainimal/data/extracted_frames/Siting',
                        blur_threshold=100.0)