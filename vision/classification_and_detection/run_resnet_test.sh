export MODEL_DIR='/root/workspace/sunghyuck/inference-compression/vision/classification_and_detection'
export DATA_DIR='/root/workspace/sunghyuck/inference-compression/vision/classification_and_detection/ILSVRC2012_img_val'
export SCENARIO="Offline"

./run_local.sh pytorch resnet50 gpu --accuracy 