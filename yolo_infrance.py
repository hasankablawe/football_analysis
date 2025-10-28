from ultralytics import YOLO
model = YOLO('/home/linux/my_new_project/ml_projects/match_analsys/Models/best.engine',task='detect')

result = model.predict('/home/linux/my_new_project/A1606b0e6_0 (10).mp4',save=True)
print(result[0])
print('_'*20)
for box in result[0].boxes:
    print(box)