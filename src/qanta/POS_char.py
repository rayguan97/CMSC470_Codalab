import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def detokenize(tokens):
	twd = TreebankWordDetokenizer()
	return twd.detokenize(tokens)

def POS_char(text):
	#divide into sentences
	sents = nltk.sent_tokenize(text)
	out = []
	indices = []
	curr = ""

	for sent in sents:
		temp = []
		#divide into tokens
		tokens = nltk.word_tokenize(sent)
		for i in nltk.pos_tag(tokens):
			temp.append(i[0])
			# If item word is a preposition, noun, or a verb.
			if i[1] == 'PRP' or 'NN' in i[1] or i[1] == 'VB':
				# Add and detokenizer sentence.
				temp_str = curr + detokenize(temp)
				out.append(temp_str)
				indices.append(len(temp_str))
			
		
		temp_str = curr + detokenize(temp)
		out.append(temp_str)	
		indices.append(len(temp_str))
		curr += detokenize(temp) + " "

	
	return out, indices