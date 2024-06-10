# XL-HQL

## Environment

The python == 3.6.9, torch== 1.7.0

~~~
pip install requirements
~~~

## DataSet

Follow previous work, we use their dataset: https://github.com/zy-zhou/HQLgen; In addition, we translate non-English comments into English via XL-HQL/hqlxlent-master/data_filter.py. You can run data_filter.py to get the processed dataset. Finally, put the handled dataset into hqlxlent-master/data_and_model and word2_vec-master/data_and_model ,respectively.

## Train

### word2vec 

If you want to use word2vec to encoding,  you can run word2vec-master/extract_vocan_hql.py to build vocabulary firstly and word_embeeding.py to get word2vec model. Last, run the train.py to train model.

~~~ 
python extract_vocab_hql.py
python word_embeeding.py
python train.py --do_train --tepoch 200 --bS 32 --accumulate_gradients 2 --bert_type_abb uS --lr 0.001 --lr_bert 0.00001 --seed 1 --num_target_layers 1
~~~

### Another Models

If you want to use xlnet/albert/gpt2 model , you install related pre-trained models. Then, Put them in the hqlxlnet-master folder. 

Mixed 

~~~
python train.py --do_train --tepoch 40 --bS 32 --accumulate_gradients 2 --bert_type_abb uS --lr 0.001 --lr_bert 0.00001 --seed 1 --num_target_layers 1 
~~~

Cross-project

~~~
python train.py --do_train --tepoch 40 --bS 32 --accumulate_gradients 2 --bert_type_abb uS --lr 0.001 --lr_bert 0.00001 --seed 1 --num_target_layers 1 --project
~~~

## Test

You can run hql_test.py to calculate the acc score. 

Mixed

~~~
python  hql_test.py  --trained
~~~

Cross-project

~~~
python  hql_test.py  --trained --project
~~~
