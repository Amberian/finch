<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/tensorflow_nlp.png" height='100'>

---

#### Contents
* NLP
    * [Word Embedding（词向量）](https://github.com/zhedongzheng/finch#word-embedding%E8%AF%8D%E5%90%91%E9%87%8F)
    * [Text Classification（文本分类）](https://github.com/zhedongzheng/finch#text-classification%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB)
    * [Text Generation（文本生成）](https://github.com/zhedongzheng/finch#text-generation%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90)
    * [Text Matching（文本匹配）](https://github.com/zhedongzheng/finch/blob/master/README.md#text-matching%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D)
    * [Sequence Labelling（序列标记）](https://github.com/zhedongzheng/finch#sequence-labelling%E5%BA%8F%E5%88%97%E6%A0%87%E8%AE%B0)
    * [Sequence to Sequence（序列到序列）](https://github.com/zhedongzheng/finch#sequence-to-sequence%E5%BA%8F%E5%88%97%E5%88%B0%E5%BA%8F%E5%88%97)
    * [Question Answering（问题回答）](https://github.com/zhedongzheng/finch/blob/master/README.md#question-answering%E9%97%AE%E9%A2%98%E5%9B%9E%E7%AD%94)
    * [Knowledge Graph（知识图谱）](https://github.com/zhedongzheng/finch#knowledge-graph%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1)
    * [TensorFlow](https://github.com/zhedongzheng/finch/blob/master/README.md#tensorflow)
    * [Spark](https://github.com/zhedongzheng/finch/blob/master/README.md#spark)
* [Speech](https://github.com/zhedongzheng/finch#speech%E8%AF%AD%E9%9F%B3)
* [Vision](https://github.com/zhedongzheng/finch#computer-vision%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89)
* [Reinforcement Learning](https://github.com/zhedongzheng/finch#reinforcement-learning%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0)


---
#### Requirements 
* Python 3 and [Jupyter Notebook](http://jupyter.org/) are required

    ```
    (CPU User) $ pip3 install tensorflow sklearn scipy bunch tqdm wget
    
    (GPU User) $ pip3 install tensorflow-gpu sklearn scipy bunch tqdm wget
    ```
---

#### Word Embedding（词向量）
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/decoration_6.png" height='100'>

 * Skip-Gram &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/word2vec_skipgram.ipynb)

 * CBOW &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/word2vec_cbow.ipynb)

#### Text Classification（文本分类）
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/decoration_2.png" height='100'>

 * TF-IDF + LR &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/tfidf_imdb_test.ipynb)

 * Text-CNN &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/concat_conv_1d_text_clf_imdb_test.ipynb)

     * Word + Char Embedding &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/char_embedding.ipynb)

     * Gated CNN &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/glu_imdb_test.ipynb)

     * ConvLSTM &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/convlstm_imdb_test.ipynb)

 * Bi-RNN &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/rnn_text_clf_imdb_test.ipynb)

 * Attention-Pooling &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/only_attn_text_clf_varlen_imdb_test.ipynb)

 * FastText &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/fasttext_imdb_test.ipynb)

#### Text Generation（文本生成）
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/decoration_5.png" height='100'>

* Language Model（语言模型）

   * RNN + Beam-Search &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-data-api/char_rnn_beam_test.ipynb)

   * Dilated-Conv + Beam-Search &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-data-api/cnn_lm_test_beam_search.ipynb)

   * Self-Attention + Beam-Search &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-data-api/self_attn_lm_test_beam_search.ipynb)
   
   * Character Aware &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-data-api/cnn_rnn_text_gen_test.ipynb)

#### Text Matching（文本匹配）
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/decoration_10.jpeg" height='100'>

* User-Item Matching &nbsp; &nbsp; [Folder](https://github.com/zhedongzheng/finch/tree/master/src_nlp/tensorflow/movielens)
    
* Question Matching &nbsp; &nbsp; [Folder](https://github.com/zhedongzheng/finch/tree/master/src_nlp/tensorflow/competition/ppdai)

#### Sequence Labelling（序列标记）
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/decoration_4.jpg" height='100'>

* POS Tagging（词性识别）

    * Bi-RNN + CRF &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-data-api/pos_birnn_crf_test.ipynb)

    * CNN + Attention + CRF &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-data-api/cnn_attn_seq_label_pos_test.ipynb)

* Chinese Segmentation（中文分词）

    * Bi-RNN + CRF &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-data-api/chseg_birnn_crf_test.ipynb)

    * CNN + Attention + CRF &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-data-api/cnn_attn_seq_label_chseg_test.ipynb)

#### Sequence to Sequence（序列到序列）
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/decoration_1.png" height='100'>

* Learning to Sort（机器排序）

    * Seq2Seq + Attention + Beam-Search &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/seq2seq_ultimate_test.ipynb)

    * Pointer Network &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/tf-estimator/pointer_net_test.ipynb)

    * Transformer &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/attn_is_all_u_need/train_letters.ipynb)
    
* Learning to Dialog（机器对话）

    * Transformer &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/attn_is_all_u_need/train_dialog.ipynb)

* VAE (Variational Autoencoder)

    * Recurrent VAE &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/vae/train.ipynb)

        * Variational Inference via ```tf.distributions``` &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/vae/train_tfd.ipynb)

    * "Toward Controlled Generation of Text" (ICML 2017) &nbsp; &nbsp; [Folder](https://github.com/zhedongzheng/finch/tree/master/src_nlp/tensorflow/toward_control)

* Data Argumentation（数据增强）

    * Back Translation &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/other/bi_translation.ipynb)
        
#### Question Answering（问题回答）
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/dmn-details.png" height='100'>

* Memory Network（记忆网络）

    *  End-to-End Memory Network &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/end2end_mn/train.ipynb)

    *  Dynamic Memory Network &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_nlp/tensorflow/dmn/train.ipynb)

#### Knowledge Graph（知识图谱）
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/kg.png" height='100'>

* Knowledge Representation（知识表示）

    * RDF + SPARQL &nbsp; &nbsp; [Notebook (WN18)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_kg/knowledge_representation/tests/wn18_rdf_sparql.ipynb)

* Link Prediction (链路预测)

    * DistMult (1-1 Scoring) &nbsp; &nbsp; [Notebook (WN18)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_kg/link_prediction/main/wn18_distmult_1v1.ipynb)

    * DistMult (1-N Scoring) &nbsp; &nbsp; [Notebook (WN18)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_kg/link_prediction/main/wn18_distmult_1vn.ipynb)

    * ConvE (1-N Scoring) &nbsp; &nbsp; [Notebook (WN18)](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_kg/link_prediction/main/wn18_conve_1vn.ipynb)

#### TensorFlow
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/tf.png" height='100'>

* Loading Data (导入数据)

    * TFRecord: Fixed Length &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_tf/data_io/tfrecord_imdb_fixed_len.ipynb)
    
    * TFRecord: Padded Batch &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_tf/data_io/tfrecord_imdb_var_len.ipynb)

* Project Template（项目模版）

    * TF Estimator Template &nbsp; &nbsp; [Folder](https://github.com/zhedongzheng/finch/tree/master/src_tf/templates/tf_estimator_template)

    * TF Dataset Template &nbsp; &nbsp; [Folder](https://github.com/zhedongzheng/finch/tree/master/src_tf/templates/tf_dataset_template)

#### Spark
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/spark.png" height='100'>

* Text Classification &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_spark/classification.ipynb)

* Topic Modelling &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_spark/topic.ipynb)

#### Speech（语音)
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/speech.png" height='100'>

* Bi-RNN + CTC &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_speech/asr/train.ipynb)

#### Computer Vision（计算机视觉)
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/vision.png" height='150'>

* DCGAN &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_cv/dcgan.ipynb)

#### Reinforcement Learning（强化学习)
<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/reinforcement.jpg" height='100'>

* Policy Gradient &nbsp; &nbsp; [Notebook](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/src_rl/pg_cartpole.ipynb)
