<img src="https://github.com/zhedongzheng/finch/blob/master/src_nlp/assets/movielens.png">

* The idea comes from [Here](http://www.paddlepaddle.org/docs/develop/book/05.recommender_system/index.html)

* We need to switch to Python 2 for this sub-project

    * Because we need to use [PaddlePaddle](http://www.paddlepaddle.org/) for processed Movielens data

        * which only supports Python 2 for now

* Please install these packages under Python 2:
    ```
    pip install paddlepaddle tensorflow pandas tqdm
    ```
* First generate data, then train the model
    ```
    $ cd ./data
    $ python make_data.py
    $ cd ..
    $ python train.py
    ```

    ```
    Testing loss: 3.7632217
    Testing loss: 3.4815586
    Testing loss: 3.5643203
    Testing loss: 3.313573
    Testing loss: 3.2979832
    Testing loss: 3.2159936
    Testing loss: 3.1644676
    Testing loss: 3.1356244
    Testing loss: 3.115319
    Testing loss: 3.0999339
    Testing loss: 3.0713046
    Testing loss: 3.0459547
    Testing loss: 3.0222063
    Testing loss: 3.0040457
    Testing loss: 3.0101979
    Testing loss: 2.9714873
    Testing loss: 3.002457
    Testing loss: 2.9743514
    Testing loss: 2.95745
    Testing loss: 2.970417
    Testing loss: 2.9601738
    Testing loss: 2.952818
    Testing loss: 2.9619434
    Testing loss: 2.9600794
    Testing loss: 2.9565902
    Testing loss: 2.9566388
    Testing loss: 2.955195
    Testing loss: 2.9507756
    Testing loss: 2.9355392
    Testing loss: 2.9538527
    Testing loss: 2.9356554
    Testing loss: 2.9418507
    Testing loss: 2.9342399
    Testing loss: 2.94277
    Testing loss: 2.933206
    Testing loss: 2.9272215
    Testing loss: 2.9307425
    Testing loss: 2.9275203
    Testing loss: 2.9275682
    Testing loss: 2.9228027
    Testing loss: 2.9249346
    Testing loss: 2.9238608
    Testing loss: 2.9257743
    Testing loss: 2.9205306
    Testing loss: 2.9220045
    Testing loss: 2.920885
    Testing loss: 2.9214213
    Testing loss: 2.9231598
    Testing loss: 2.9181385
    Testing loss: 2.928637
    ```
