wget https://msraprophetnet.blob.core.windows.net/prophetnet/release_checkpoints/prophetnet_multi.pt

mkdir squad/tokenized
mkdir squad/processed
python xprophetnet_tokenize_squad.py

fairseq-preprocess \
--user-dir ./prophetnet \
--task translation_prophetnet \
--source-lang src --target-lang tgt \
--trainpref ./squad/tokenized/en.train --validpref ./squad/tokenized/en.dev --testpref ./squad/tokenized/en.test \
--destdir ./squad/processed --srcdict prophetnet_multi_dict/dict.txt --tgtdict prophetnet_multi_dict/dict.txt \
--workers 15
