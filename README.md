# Kesci 水下目标检测算法赛  underwater object detection algorithm contest Baseline <font color=red>**A榜 mAP 46-47**</font><br /> 

## 比赛地址：[Kesci 水下目标检测](https://www.kesci.com/home/competition/5e535a612537a0002ca864ac)

## Update 更新使用htc预训练的resnext101 64x4d 线上mAP为**48.7**

## 整体思路
   + detection algorithm: Cascade R-CNN 
   + backbone: ResNet50 + FPN
   + post process: soft nms
   + 基于[mmdetection](https://github.com/open-mmlab/mmdetection/), 不是最新版，大家可以自己升级
   + res50 和se50 均可以达到线上testA 46-47 mAP, 经过[spytensor](https://github.com/spytensor)验证集成下可以48-49
   + resnext101 64x4d 48.7mAP
## 代码环境及依赖

+ OS: Ubuntu16.10
+ GPU: 2080Ti * 4
+ python: python3.7
+ nvidia 依赖:
   - cuda: 10.0.130
   - cudnn: 7.5.1
   - nvidia driver version: 430.14
+ deeplearning 框架: pytorch1.1.0
+ 其他依赖请参考requirement.txt
+ 显卡数量不太重要，大家依据自身显卡数量倍数调整学习率大小即可

## 训练数据准备

- **相应文件夹创建准备**

  - 在代码根目录下新建data文件夹，或者依据自身情况建立软链接
  - 进入data文件夹,创建文件夹:
  
     annotations

     pretrained

     results

     submit

  - 将官方提供的训练和测试数据解压到data目录中，产生：
    
    train

    test-A-image
    
    
- **label文件格式转换**

  - 官方提供的是VOC格式的xml类型label文件，个人习惯使用COCO格式用于训练，所以进行格式转换
  
  - 使用 tools/data_process/xml2coco.py 将label文件转换为COCO格式，新的label文件 train.json 会保存在 data/train/annotations 目录下

  - 为了方便利用mmd多进程测试（速度较快），我们对test数据也生成一个伪标签文件,运行 tools/data_process/generate_test_json.py 生成 testA.json, 伪标签文件会保存在data/train/annotations 目录下

  - 总体运行内容：

    - python tools/data_process/xml2coco.py

    - python tools/data_process/generate_test_json.py

- **预训练模型下载**
  - 下载mmdetection官方开源的casacde-rcnn-r50-fpn-2x的COCO预训练模型[cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth](https://open-mmlab.oss-cn-beijing.aliyuncs.com/mmdetection/models/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth)并放置于 data/pretrained 目录下
  - senet50的预训练详见: [mmd-senet](https://github.com/zhengye1995/pretrained), 这里要特别感谢[jsonc](https://github.com/jsnoc) 大佬提供的预训练模型
  - 下载mmdetection官方开源的htc的[resnext 64x4d 预训练模型](https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth)

## 依赖安装及编译


- **依赖安装编译**

   1. 创建并激活虚拟环境
        conda create -n underwater python=3.7 -y
        conda activate underwater

   2. 安装 pytorch
        conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch
        
   3. 安装其他依赖
        pip install cython && pip --no-cache-dir install -r requirements.txt
   
   4. 编译cuda op等：
        python setup.py develop
   

## 模型训练及预测
    
   - **训练**

	1. 运行：
        
        r50:
        
		chmod +x tools/dist_train.sh

        ./tools/dist_train.sh configs/underwater/cas_r50/cascade_rcnn_r50_fpn_1x.py 4
        
        se50:
        
		chmod +x tools/dist_train.sh

        ./tools/dist_train.sh configs/underwater/cas_se/cas_se50_12ep.py 4
        
        x101_64x4d (htc pretrained):
        
		chmod +x tools/dist_train.sh

        ./tools/dist_train.sh configs/underwater/cas_x101/cascade_rcnn_x101_64x4d_fpn_1x.py 4
        
        (上面的4是我的gpu数量，请自行修改)

   	2. 训练过程文件及最终权重文件均保存在config文件中指定的workdir目录中

   - **预测**

    1. 运行:
    
        r50:

        chmod +x tools/dist_test.sh

        ./tools/dist_test.sh configs/underwater/cas_r50/cascade_rcnn_r50_fpn_1x.py workdirs/cascade_rcnn_r50_fpn_1x/latest.pth 4 --json_out results/cas_r50.json

        (上面的4是我的gpu数量，请自行修改)
        
        se50:

        chmod +x tools/dist_test.sh

        ./tools/dist_test.sh configs/underwater/cas_se/cas_se50_12ep.py workdirs/cas_se50_12ep/latest.pth 4 --json_out results/cas_se50.json

        (上面的4是我的gpu数量，请自行修改)
        
        x101_64x4d (htc pretrained):

        chmod +x tools/dist_test.sh

        ./tools/dist_test.sh configs/underwater/cas_se/cascade_rcnn_x101_64x4d_fpn_1x.py workdirs/cas_x101_64x4d_fpn_htc_1x/latest.pth 4 --json_out results/cas_x101.json


    2. 预测结果文件会保存在 /results 目录下

    3. 转化mmd预测结果为提交csv格式文件：
       
       python tools/post_process/json2submit.py --test_json cas_r50.bbox.json --submit_file cas_r50.csv
       
       python tools/post_process/json2submit.py --test_json cas_se50.bbox.json --submit_file cas_se50.csv
       
       python tools/post_process/json2submit.py --test_json cas_x101.bbox.json --submit_file cas_x101.csv

       最终符合官方要求格式的提交文件 cas_r50.csv, cas_se50.csv 和 cas_x101.csv 位于 submit目录下
    

## Contact

    author：rill

    email：18813124313@163.com
