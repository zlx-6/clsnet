## Introduction
😃😃😃This is a reimplement classification networks based on the [MMClassification](https://github.com/open-mmlab/mmclassification) and [MMCV](https://github.com/open-mmlab/mmcv). I will combine the mmcv and mmclasification in this repo. You only need install base pytorch enviroment and don't install and complie the MMCV and MMClassification in your computer which will aviod much complie probelms. But this repo don't support the deforconv. 

I will add some networks as follows:

- [x] MobileNetV3
- [ ] MobileNetV2
- [ ] VGG
- [ ] ResNet
- [ ] ResNeXt
- [ ] RepVGG
- [ ] ViT

Because the resources limition, this repo will train the model on the cifar10.

## Start
You should clone this repositories and run the follow code:

    git clone https://github.com/zlx-6/clsnet.git
    cd clsnet
    pip install -r requirements.txt

## Train a model
 Trian a model from the screath 
    
    python main.py --config config_file_path --work-dir save_pth_and_log_dir --resume-from checkpoint_file_pth --device device --gpus gpu_ids

- if you want train a new model from the pretrained the model, you can set the load_from ="pth_path" in the config file.
- you can change the lr and bs in you config file
  
        runner = dict(type='EpochBasedRunner', max_epochs=12),
        data = dict(samples_per_gpu=2,workers_per_gpu=2,)


