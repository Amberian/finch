[Download Data](https://pan.baidu.com/s/1uXY2oY5s_EFSuQEznbvmBQ)

```
├── configs
│   └── general.py   
│
├── data               
│   └── dataloaders
│   	 └── dataloader_char_rnn.py
│   	 └── dataloader_word_fixed.py
│   	 └── dataloader_word_rnn.py
│   	 └── preprocess_char_rnn.py
│   	 └── preprocess_word_fixed.py
│   	 └── preprocess_word_rnn.py
│   └── files_original
│   	 └── char_embed.txt        # download and place here
│   	 └── question.csv          # download and place here
│   	 └── test.csv              # download and place here
│   	 └── train.csv             # download and place here
│   	 └── word_embed.txt        # download and place here
│   └── files_processed
│   └── notebooks
│   └── tfrecords
│
├── log             
│   └── example.py
│   └── 2018-07-02 06:22:36.log    # example of training
│   
├── mains              
│   └── train_baseline_word.py     # run this to train model
│  
└── model  
    └── baseline.py
```
