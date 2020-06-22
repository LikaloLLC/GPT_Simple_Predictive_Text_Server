

## Quick Start

1. download GPT2 pre-trained model in Pytorch which huggingface/pytorch-pretrained-BERT already made! (Thanks for sharing! it's help my problem transferring tensorflow(ckpt) file to Pytorch Model!)
```shell
$ git clone https://github.com/graykode/gpt-2-Pytorch && cd gpt-2-Pytorch
# download huggingface's pytorch model 
$ curl --output gpt2-pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
# setup requirements, if using mac os, then run additional setup as descibed below
$ pip install -r requirements.txt
```

