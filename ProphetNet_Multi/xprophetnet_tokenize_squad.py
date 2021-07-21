import sentencepiece as spm
import os

sp = spm.SentencePieceProcessor()
sp.Load("prophetnet_multi_dict/sentencepiece.bpe.model")

#train.pa.txt
dirs = os.listdir('squad/raw')
print(dirs)
for file_name in dirs:
	f = open('squad/raw/{}'.format(file_name), 'r', encoding='utf-8')
	file_name_part = file_name.split('.')
	file_out_name = '{}.{}.{}'.format('en', file_name_part[0],file_name_part[1])
	fout = open('squad/tokenized/{}'.format(file_out_name), 'w', encoding='utf-8')
	for line in f:
		tok = sp.EncodeAsPieces(line.strip())[:256]
		fout.write('{}\n'.format(" ".join(tok)))
f.close()
fout.close()