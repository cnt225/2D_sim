# Riemannian Conditional FlowMatching

## Requirements
### Environment
For Linux,
```shell
conda create -n fm python=3.10
conda activate fm
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
```

For Windows,
```shell
conda create -n fm python=3.10
conda activate fm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Run
```shell
python train.py --config ./configs/grasp_rcfm.yml --device 0 --logdir results --run test
```
