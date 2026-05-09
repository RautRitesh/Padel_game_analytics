from ultralytics import YOLO 
#new ball detection yolo model
model = YOLO(r"../models/best.pt")
result= model.track(r'../input_videos/infernce_sample_video.mp4',conf=0.55,
                    save=True,
                    project=r'C:\Users\RiteshRaut\Yolo\Padel_game_analytics\output_videos',
                    name='detect')

print("sucessfull completion")