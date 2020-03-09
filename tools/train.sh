#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/jiuping/cascade_mask_rcnn_r50_fpn_1x.py 4
python -m torch.distributed.launch --nproc_per_node=4 tools/test.py configs/jiuping/cascade_mask_rcnn_r50_fpn_1x.py work_dirs/cascade_mask_rcnn_r50_fpn_1x_1000_bs3/latest.pth --json_out results/cas_r50_mask_800_1000_bs3.json --launcher pytorch --eval bbox
python tools/post_process/json2submit.py --test_json cas_r50_mask_800_1000_bs3.bbox.json --submit_file cas_r50_mask_800_1000_bs3.json
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/jiuping/cascade_rcnn_r50_fpn_1x.py 4
#python -m torch.distributed.launch --nproc_per_node=4 tools/test.py configs/jiuping/cascade_rcnn_r50_fpn_1x.py work_dirs/cascade_rcnn_r50_fpn_1x_giou/latest.pth --json_out results/cas_r50_giou.json --launcher pytorch --eval bbox
#python tools/post_process/json2submit.py --test_json cas_r50_giou.bbox.json --submit_file cas_r50_giou.json
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/jiuping/cascade_rcnn_seresnext50_fpn_1x.py 4
#python -m torch.distributed.launch --nproc_per_node=4 tools/test.py configs/jiuping/cascade_rcnn_seresnext50_fpn_1x.py work_dirs/cascade_rcnn_seresnext50_1x/latest.pth --json_out results/cas_se50.json --launcher pytorch --eval bbox
#python tools/post_process/json2submit.py --test_json cas_se50.bbox.json --submit_file cas_se50.json
#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/jiuping/cascade_rcnn_r50_fpn_1x_bboxjitter.py 4
#python -m torch.distributed.launch --nproc_per_node=4 tools/test.py configs/jiuping/cascade_rcnn_r50_fpn_1x_bboxjitter.py work_dirs/cascade_rcnn_r50_fpn_1x_bboxjitter/latest.pth --json_out results/cas_r50_add_bboxjitter.json --launcher pytorch --eval bbox
#python tools/post_process/json2submit.py --test_json cas_r50_add_bboxjitter.bbox.json --submit_file cas_r50_add_bboxjitter.json

#CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh configs/jiuping/cascade_rcnn_r34_fpn_1x.py 4
#python -m torch.distributed.launch --nproc_per_node=4 tools/test.py configs/jiuping/cascade_rcnn_r34_fpn_1x.py work_dirs/cascade_rcnn_r34_fpn_1x/latest.pth --json_out results/cas_r34.json --launcher pytorch --eval bbox
#python tools/post_process/json2submit.py --test_json cas_r34.bbox.json --submit_file cas_r34.json
