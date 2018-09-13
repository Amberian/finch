Implementing the idea of [Toward Controlled Generation of Text](https://arxiv.org/abs/1703.00955?context=cs)

<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/control-vae.png" height='300'>


```
├── base
│   └── base_model.py   
│   └── base_trainer.py   
│
├── configs
│   └── config.py   
│
├── data               
│   └── imdb        
│       └── vae_pipeline.py
│       └── wake_sleep_pipeline.py
│       └── discriminator_pipeline.py
│
├── log             
│   └── example.py
│   └── 2018-06-06 22:49:48.log   # sample of training process
│
├── mains              
│   └── pretrain.py               # first run this
│   └── train.py                  # then run this
│  
├── model              
│   └── discriminator.py
│   └── encoder.py
│   └── generator.py
│   └── vae.py
│   └── wake_sleep.py
│
├── trainer              
│   └── vae_trainer.py
│   └── wake_sleep_trainer.py
│
├── utils             
│   └── modified.py
│
├── vocab
│   └── imdb
│       └── imdb_vocab.py
```
