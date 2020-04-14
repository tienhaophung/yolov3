# Training
python train.py --cfg cfg/yolov3-spp-custom.cfg --data data/custom/custom.data --weights weights/yolov3-spp-ultralytics.pt --single-cls --device 0 --epochs 50 --batch-size 2 --adam
# Testing
python test.py --cfg cfg/yolov3-spp-custom.cfg --data data/custom/custom.data --weights weights/best.pt --single-cls --batch-size 8 --setname test
# Detect
python detect.py --cfg cfg\yolov3-spp-custom.cfg --names data/custom/classes.names --weights weights/best.pt --source C:\Users\User\Desktop\simple-HRNet\hand_images --img-size 416