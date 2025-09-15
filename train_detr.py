from ultralytics import RTDETR

model = RTDETR('rt-detr/rtdetr-l.pt')

print(model.info())
data = "data_detr.yaml"

res = model.train(data=data, epochs=50, imgsz=640, save=True, plots=True, device='0,1,2,3,4,5,6,7', workers=72, weight_decay=1e-3)

