# f
```bash
#Pretrain the model
!python src/run.py pretrain vanilla wiki.txt \
--writing_params_path vanilla.pretrain.params
```
out:
```bash
epoch 1 iter 22: train loss 3.46976. lr 5.999655e-03: 100%|█| 23/23 [00:03<00:00
epoch 2 iter 22: train loss 3.15410. lr 5.998582e-03: 100%|█| 23/23 [00:02<00:00
epoch 3 iter 22: train loss 2.98234. lr 5.996780e-03: 100%|█| 23/23 [00:02<00:00
epoch 4 iter 22: train loss 2.89906. lr 5.994250e-03: 100%|█| 23/23 [00:02<00:00
epoch 5 iter 22: train loss 2.82379. lr 5.990993e-03: 100%|█| 23/23 [00:02<00:00
...
...
...
epoch 649 iter 22: train loss 0.47791. lr 6.881640e-04: 100%|█| 23/23 [00:02<00:
epoch 650 iter 22: train loss 0.50068. lr 7.182453e-04: 100%|█| 23/23 [00:02<00:
```
--------
```bash
# Finetune the model
!python src/run.py finetune vanilla wiki.txt \
--reading_params_path vanilla.pretrain.params \
--writing_params_path vanilla.finetune.params \
--finetune_corpus_path birth_places_train.tsv
```
out:
```bash
epoch 1 iter 7: train loss 0.72708. lr 5.999844e-04: 100%|█| 8/8 [00:02<00:00,  
epoch 2 iter 7: train loss 0.56959. lr 5.999351e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 3 iter 7: train loss 0.48298. lr 5.998521e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 4 iter 7: train loss 0.41198. lr 5.997352e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 5 iter 7: train loss 0.33263. lr 5.995847e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 6 iter 7: train loss 0.30696. lr 5.994004e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 7 iter 7: train loss 0.24358. lr 5.991823e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 8 iter 7: train loss 0.19598. lr 5.989306e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 9 iter 7: train loss 0.16647. lr 5.986453e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 10 iter 7: train loss 0.14955. lr 5.983263e-04: 100%|█| 8/8 [00:01<00:00, 
```
----------
```bash
# Evaluate on the dev set; write to disk
!python src/run.py evaluate vanilla wiki.txt \
--reading_params_path vanilla.finetune.params \
--eval_corpus_path birth_dev.tsv \
--outputs_path vanilla.pretrain.dev.predictions
```
out:
```bash
data has 418352 characters, 256 unique.
number of parameters: 3323392
500it [00:37, 13.44it/s]
Correct: 115.0 out of 500.0: 23.0%
```
-------------
```bash
# Evaluate on the test set; write to disk
!python src/run.py evaluate vanilla wiki.txt \
--reading_params_path vanilla.finetune.params \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path vanilla.pretrain.test.predictions
```
out:
```bash
data has 418352 characters, 256 unique.
number of parameters: 3323392
437it [00:31, 13.70it/s]
No gold birth places provided; returning (0,0)
Predictions written to vanilla.pretrain.test.predictions; no targets provided
```
# g
pretraining
```bash
epoch 648 iter 22: train loss 0.53710. lr 6.586443e-04: 100%|█| 23/23 [00:02<00:
epoch 649 iter 22: train loss 0.55729. lr 6.881640e-04: 100%|█| 23/23 [00:02<00:
epoch 650 iter 22: train loss 0.50611. lr 7.182453e-04: 100%|█| 23/23 [00:02<00:
```
finetuning
```bash
epoch 1 iter 7: train loss 0.77864. lr 5.999844e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 2 iter 7: train loss 0.64808. lr 5.999351e-04: 100%|█| 8/8 [00:01<00:00,  
... 
epoch 9 iter 7: train loss 0.30447. lr 5.986453e-04: 100%|█| 8/8 [00:01<00:00,  
epoch 10 iter 7: train loss 0.26082. lr 5.983263e-04: 100%|█| 8/8 [00:01<00:00, 
```
evaluate
```bash
device is  0
data has 418352 characters, 256 unique.
number of parameters: 3076988
500it [00:36, 13.68it/s]
Correct: 66.0 out of 500.0: 13.200000000000001%
```