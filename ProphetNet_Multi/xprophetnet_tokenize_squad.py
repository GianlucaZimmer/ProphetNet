import sentencepiece as spm
import os

sp = spm.SentencePieceProcessor()
sp.Load("prophetnet_multi_dict/sentencepiece.bpe.model")

#en.train.src.txt
dirs = os.listdir('squad/raw')
print(dirs)
for file_name in dirs:
	f = open('squad/raw/{}'.format(file_name), 'r', encoding='utf-8')
	file_name_part = file_name.split('.')
	#en.train.src
	file_out_name = '{}.{}.{}'.format(file_name_part[0], file_name_part[1],file_name_part[2])
	fout = open('squad/tokenized/{}'.format(file_out_name), 'w', encoding='utf-8')
	if "src" in file_name:
		for line in f:
			org = line.strip().split("[SEP]")
			ans = sp.EncodeAsPieces(org[1].strip())
			par = sp.EncodeAsPieces(org[0].strip())[:510 - len(ans)] # max 512 tokens
			fout.write('{} [SEP] {}\n'.format(" ".join(ans), " ".join(par)))
	else:
		for line in f:
			org = sp.EncodeAsPieces(line.strip())
			fout.write('{}\n'.format(" ".join(org)))
f.close()
fout.close()