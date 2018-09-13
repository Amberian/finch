<img src="https://github.com/zhedongzheng/finch/blob/master/nlp-models/assets/dmn-details.png">

---

* Many functions are adapted from [Alex Barron's](https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow) work, thanks for him!
* Based on that:
    * We have used ```tf.estimator.Estimator``` API to package the model
    * We have used ```tf.map_fn``` to replace the Python for loop, which makes the model truly dynamic
    * We have added a decoder in the answer module for "talking"
    * We have reproduced ```AttentionGRUCell``` from new official ```GRUCell```
