from ultralytics import YOLO 
model = YOLO("models/best.pt")
result= model.track('input_vidoes/infernce_sample_video.mp4',
                    project=r'C:\Users\RiteshRaut\Yolo\Padel_game_analytics\output_videos',
                    name='detect')

print("sucessfull completion")