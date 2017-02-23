from gensim.models import Word2Vec, Phrases
import gensim.models.phrases as gmp
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.tag import PerceptronTagger
import numpy as np
from pympler.asizeof import asizeof
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from utils import *
import tensorflow as tf
sess = tf.Session()
import cPickle as pickle, os, string, re, sys, math, random

#################################
##### PACKAGE REQUIREMENTS ######
#################################
# Gensim, Keras, Numpy, NLTK, 	#
# MatPlotLib, h5py, Cython		#
# TensorFlow (<= 1.0)			#
#								#
# In NLTK, you only need the 	#
# Punkt and PerceptronTagger	#
# datasets and the stopwords 	#
#################################

#################################################################################################
#############################      (CONTEXT PREDICTED WORDS).py     #############################
#############################                 USAGE                 #############################
#################################################################################################
# Run this program from the command line with Python 2.7									  	#
# Natively, variables are configured for the BitterLemons dataset.								#
#																								#
# There are three important flags:																#
#	--generate  : which generates new training data and labels from a list of documents.		#
#	--train     : which creates a new neural network and trains on your generated data 			#
#	--predict   : which predicts a list of keywords for each document in your network.			#
#																								#
# You must generate before you can train, and train before you can predict. Unless				#
# specified (by changing the paths), retraining will overwrite past data.												#
#																								#
# Additionally, there are several other flags:													#
#	--difficult : which prints the least-predictive words in addition to the most				#
#	--intersect : which prints the (soft) intersection of keywords in your document 			#
#	--continue  : which loads a neural network from disk and continues training it 				#
#	--SemEval   : which switches all paths to be correct for the SemEval2010 dataset 			#
#   --piazza    : which switches all paths to be correct for the PiazzaBot project  			#
#	--verbose   : which prints individual document keywords even if printing the intersection 	#
#																								#
# Finally, there are a number of parameters you can pass in. 									#
# The syntax is "--<key>:<value>". This is a list of valid keys: 								#
#	outfile 	: the value corresponds to the path to documents 								#
#	model 		: the value corresponds to the path to the keras model 							#
#	cbow 		: the value corresponses to the path to generate data to 						#
#	epochs 		: the value is the number of epochs to train for 								#
#	documents	: the value is the number of documents to predict keywords for 					#
#	keywords 	: the value is the number of keywords to predict 								#
# 																								#
# If you stumble upon this somehow, please feel free to use. However, if your work 				#
# leads to any publication or distribution, I would appreciate proper attribution. 				#
#																								#
#################################################################################################


# This is just a convenience variable
DATASET_PATH = "../datasets/bt.1.0/"

logs_path = mkdir("../tensorflow_logs") + "/example"

# This is a path to a pickled file containing one thing: a list of documents in string form.
OUTFILE = DATASET_PATH + "bitterlemons_files.p"

# This is a path to a pickled file containing four things in this order:
#	1) The context word vectors one-hot-encoded
#	2) The missing center word for each vector in part 1, also one-hot-encoded
#	3) The number of words encountered
#	4) The missing center word for each vector as its original string (or potential Bi/trigram)
#	5) A list detailing the number of context-window vectors each document contains in order
#
#	Note: Potentially this has not been created yet and will be created with the --generate flag
CBOW_PATH = DATASET_PATH + "CBOW.p"

# This is the maximum size (in megabytes) which the --generate flag will create for a single array.
# If --generate wants to create something larger, it will split it up into multiple files in a 
# newly created "cbow_files" folder.
MAX_NBYTES = 80

# This is a path to a h5py saved version of the Keras Neural Network (topology + weights)
# Like the above, may not be created if you plan on giving the --train flag
MODEL_PATH = DATASET_PATH + "keras.model"

# How big your context window will be
WINDOW_SIZE = 9

# This is the dimension of the embedded projection
dim_proj = 250

# The dimension of the one-hot-encoding. This is typically less than the vocabulary size (which is naively ~50,000)
# In practice, this tends not to matter since strings are hashed well
max_num_words = 20000

# The starting learning rate of the neural network
lr = 1e-3

# The batch size of the neural network. Limited experimentation suggests 128 as a good value.
BATCH_SIZE = 256

# The number of batch-epochs to train for. Good results can be seen as early as 20-30
# Between 300-1000 is ideal for a baseline estimate 
NUM_EPOCH = 10

# The number of documents to predict keywords for if using the --predict flag
NUM_DOCS = None

# The number of keywords to print if using the --predict flag
NUM_KEYWORDS = 10

# If this is true, print the least-predictive words in addition to the most predictive
SHOW_LEAST_PREDICTIVE = False

# If --verbose is flagged, print individual document keywords, even if generating an intersection
# This keyword has the double-usage of printing updates per epoch if used with the --train flag
VERBOSE = False

PRINT_PATH = None

# If REGENERATE is flagged true, this program will recreate the data stored in CBOW_PATH based on the documents in DATASET_PATH
# 	Otherwise, it will load it from disk.
# If TRAIN is flagged true, this program will recreate and retrain a neural network from scratch
#	Otherwise, it will load it from disk.
# If PREDICT is flagged true, this program will load a neural network from disk and predict keywords
#	For the first NUM_DOCS documents.
# If CONTINUE is flagged true, this program will load a neural network from disk and continue training that network.
REGENERATE, TRAIN, PREDICT, CONTINUE, INTERSECT, PRINT = False, False, False, False, False, False


TESTING = True

# Code for setting all the above values to be correct. 
# I find this busy, but more conceptually nice than argparse.
# For basic purposes, --generate, --train and --predict are all you need
if __name__ == '__main__':
	for s in sys.argv[1:]:
		if s[:2] != "--":
			print("Invalid command line input.")
			quit()
		else:
			command = s[2:]
			if command == "generate":
				REGENERATE = True 
			elif command == "train":
				TRAIN = True 
			elif command == "predict":
				print("Warning: Currently there is an unresolved bug in this feature, you have been warned.")
				PREDICT = True
			elif command == "continue":
				CONTINUE = True
			elif command == "intersect":
				INTERSECT = True
			elif command == "verbose":
				VERBOSE = True
			elif command == "print":
				PRINT = True
			elif command == "SemEval":
				DATASET_PATH = "../datasets/SemEval2010/"
				OUTFILE = DATASET_PATH + "train.data.p"
				CBOW_PATH = DATASET_PATH + "CBOW.p"
				MODEL_PATH = DATASET_PATH + "keras.model"
			elif command == "piazza":
				DATASET_PATH = "../../piazza_bot/processed_information/"
				OUTFILE = DATASET_PATH + "for_cpw.p"
				CBOW_PATH = DATASET_PATH + "cbow_files/"
				MODEL_PATH = DATASET_PATH + "keras.model"
				PRINT_PATH = DATASET_PATH + "keywords.txt"
			elif command == "difficult":
				SHOW_LEAST_PREDICTIVE = True

			# Probably good to note I use ":" as a key-value delimiter rather than "=" because I think it looks better.

			elif len(command.split(":")) > 1:
				name, val = command.split(":")
				if name == "outfile":
					OUTFILE = val
				elif name == "model":
					MODEL_PATH = val
				elif name == "epochs":
					NUM_EPOCH = int(val)
				elif name == "cbow":
					CBOW_PATH = val
				elif name == "documents":
					NUM_DOCS = int(val) 
				elif name == "keywords":
					NUM_KEYWORDS = int(val)
				elif command == "printTo":
					PRINT_PATH = val
				else:
					print("Invalid command line input.")
					quit()
			else:
				print("Invalid command line input.")
				quit()

if PREDICT and CONTINUE:
	print("Invalid combination of flags.")
	quit()

# This section triggers if you supply the --generate flag, indicating 
# you want to recreate the training data/labels
if REGENERATE:

	print("Generating data from scratch.")

	texts = pickle.load(open(OUTFILE, 'rb'))[0]


	# This splits your list of texts into a list of sentences
	# At this point (in the training data) document borders
	# are removed.

	sentences = [item for text in texts for item in PunktSentenceTokenizer().tokenize(text.decode("utf8"))]
	sentences = [i.strip(' \n,.;:').replace('\n', ' ').split(' ') for i in sentences]

	# Create and train bigram/trigram converters
	unigram = Phrases(sentences, threshold=float("inf"))
	unigrams = unigram.export_phrases(sentences)

	grams = []#[gmp.Phraser(unigram)]

	sentences_copy = sentences

	threshold = 8.0

	print("Beginning multi-gram accumulation.")

	# I want to keep making larger and larger n-grams until I think
	# there are no more to be made.
	while True:
		bigram = Phrases(sentences_copy, threshold=threshold)
		bigrams = bigram.export_phrases(sentences_copy)
		z = list(set(bigrams) - set(unigrams))
		if len(z) == 0 or threshold > 12:
			break
		else:
			gram_bigram = gmp.Phraser(bigram)
			sentences_copy = gram_bigram[sentences_copy]
			unigrams = bigrams
			grams.append(gram_bigram)
			threshold += 1

	# Maybe there's a more elegant solution to this, but this alters
	# the Keras code in a minimal way.
	def gram_er(sentences):
		temp = sentences
		for g in grams:
			temp = g[temp]
		return temp

	num_words = len(set([i for k in sentences for i in k]))

	# Convert your texts into one-hot-encoded form
	# Sequences is used for keeping track of the true
	# missing word for every context

	one_hot_sentences = [bigram_one_hot(i, max_num_words, gram_er) for i in texts]
	sequences = [bigram_text_to_word_sequence(i, gram_er) for i in texts]

	doc_lens = [0]#[len(i) for i in one_hot_sentences]

	#### OUTDATED ####
	# Iterate over all your one-hot-encoded sentences
	# At each iteration, generate a WINDOW_SIZE-width
	# context window. Pop out the center word and store
	# it in labels (and the string-version in missing_words)
	# and store the context in data

	print("Producing (input, output) pairs.")

	data_labels_words = []
	num_split_up = 0
	i = 0
	for text in one_hot_sentences:
		this_length = 0
		for context_word in range(len(text)):
			start = max(0, context_word - WINDOW_SIZE/2)
			end = min(len(text), context_word + WINDOW_SIZE/2)
			temp = text[start:end]
			temp_words = sequences[i][start:end]
			missing_word = temp_words[context_word - start]
			label = temp[context_word - start]
			temp = temp[:context_word - start] + temp[context_word - start+1:]
			for word in temp:
				this_length += 1
				data_labels_words.append((word, label, missing_word))
		doc_lens.append(this_length)
		if len(data_labels_words) != 0 and sys.getsizeof(data_labels_words[0]) * len(data_labels_words) >= (1.5) * MAX_NBYTES * 1000000:

			print("Saving batch {0} to disk.".format(num_split_up))

			toSave = mkdir(DATASET_PATH + "/cbow_files/")
			with open(toSave + "CBOW_{0}.p".format(num_split_up), "wb") as f:
				pickle.dump([data_labels_words], f)
			num_split_up += 1
			data_labels_words = []
		i += 1

	# Save the results to disk
	print("Saving final batch and information to disk.")

	toSave = mkdir(DATASET_PATH + "/cbow_files/")
	with open(toSave + "CBOW_{0}.p".format(num_split_up), "wb") as f:
		pickle.dump([data_labels_words], f)
	with open(DATASET_PATH + "misc_info.p", "wb") as f:
		pickle.dump([max_num_words, doc_lens], f)
	num_words = max_num_words


# If you chose not to regenerate, instead you
# want to load your training data from disk
# It would probably be good to check to see if we
# actually want this data before loading it, but all 
# uses of the program currently do.
else:

	print("Loading data.")

	with open(DATASET_PATH + "misc_info.p", "rb") as f:
		num_words, doc_lens = pickle.load(f)


# This section triggers if you supply the --train flag
# of if you supply the --continue flag.
# This means you want to train a neural network
if TRAIN or CONTINUE or PREDICT:

	print("Creating network topology and compiling.")

	network_input = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

	embeddings = tf.Variable(tf.random_uniform([num_words, dim_proj], -1.0, 1.0))
	e = tf.nn.embedding_lookup(embeddings, network_input)

	# Purely a guess
	num_sampled = 1

	nce_weights = tf.Variable(tf.truncated_normal([num_words, dim_proj], stddev=1.0 / math.sqrt(dim_proj)))
	nce_biases = tf.Variable(tf.zeros([num_words]))

	labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
	o = tf.nn.nce_loss(nce_weights, nce_biases, e, labels, num_sampled, num_words)
	loss = tf.reduce_mean(o)

	if TESTING:
		tf.summary.scalar("loss", loss)
		merged_summary_op = tf.summary.merge_all()

	learning_rate = tf.placeholder(tf.float32, shape=(), name="LR")
	optimizer = tf.train.AdamOptimizer(learning_rate)
	sgd = optimizer.minimize(loss)

if TRAIN or CONTINUE:

	init_op = tf.global_variables_initializer()
	with sess.as_default():
		init_op.run()

		if TESTING:
			summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		if CONTINUE:

			print("Loading already-trained weights.")

			with open(MODEL_PATH, "rb") as f:
				n_w, n_b, n_e = pickle.load(f)
			assignments = [tf.assign(embeddings, n_e), tf.assign(nce_weights, n_w), tf.assign(nce_biases, n_b)]
			sess.run(assignments)

		generator = data_generator(CBOW_PATH, BATCH_SIZE)
		save_every = 5000

		print("Training model.")
		T.silence()

		for epoch in np.arange(NUM_EPOCH):
			T.tic("Generator loading")
			input_words, output_words = generator.next()
			feed_dict = {network_input: input_words, labels: output_words, learning_rate: lr}
			T.toc()
			T.tic("Running sgd.")
			if TESTING:
				_, summary = sess.run([sgd, merged_summary_op], feed_dict)
				summary_writer.add_summary(summary, epoch)
			else:
				sess.run(sgd, feed_dict)
			T.toc()
			if epoch == 0:
				clean_print("Finished Epoch 1.")
			else:
				restart_line()
				clean_print("Finished Epoch {0}".format(epoch+1))
			if epoch % save_every == 0 and epoch != 0:
				with open(MODEL_PATH, "wb") as f:
					pickle.dump([nce_weights.eval(), nce_biases.eval(), embeddings.eval()], f)

		with open(MODEL_PATH, "wb") as f:
					pickle.dump([nce_weights.eval(), nce_biases.eval(), embeddings.eval()], f)
	print("")

# This section triggers if you supply the --predict flag, meaning
# that you want to predict keywords for some documents
if PREDICT:

	keywords = []
	data_labels_words = []
	init_op = tf.global_variables_initializer()
	with sess.as_default():
		init_op.run()
		with open(MODEL_PATH, "rb") as f:
			n_w, n_b, n_e = pickle.load(f)
		assignments = [tf.assign(embeddings, n_e), tf.assign(nce_weights, n_w), tf.assign(nce_biases, n_b)]
		sess.run(assignments)

	tagger = PerceptronTagger()

	# Instantiate a dictionary mapping words to their frequency in standard English
	word_frequency = parse_frequency()

	# For each of the first NUM_DOCS documents
	#for doc in range(2, NUM_DOCS + 2):
	for doc in range(1, (NUM_DOCS + 1) if NUM_DOCS else len(doc_lens) - 1):

		T.silence()

		# Load the context vectors, label vectors and label strings for the relevant document
		doc_start, doc_end = sum(doc_lens[:doc]), sum(doc_lens[:doc+1])
		
		j = -1

		seen = 0
		while doc_start > len(data_labels_words) + seen or len(data_labels_words) == 0:
			j += 1
			seen += len(data_labels_words)
			with open(CBOW_PATH, "rb") as f:
				data_labels_words = pickle.load(f)[0]
		first_data, first_labels, first_words = np.zeros((doc_lens[doc])), np.zeros((doc_lens[doc])), doc_lens[doc] * [""]
		for _iter in range(doc_lens[doc]):
			n_d, n_l, n_w = data_labels_words[doc_start + _iter]
			first_data[_iter] = n_d
			first_labels[_iter] = n_l
			first_words[_iter] = n_w

		def batch_generator(d, l, BATCH_SIZE):
			i = 0
			while i < d.shape[0]:
				n_d, n_l = np.zeros((BATCH_SIZE)), np.zeros((BATCH_SIZE))
				end = min(i+BATCH_SIZE, d.shape[0])
				n_d[:end-i] = d[i:end]
				n_l[:end-i] = l[i:end]
				i += BATCH_SIZE
				yield n_d, n_l[:, np.newaxis]

		#### OUTDATED ####
		# This keeps track of the total loss for each target word encountered 
		# In the form:
		# losses[(str_form_of_word)] = (sum_of_loss_for_word, num_of_times_word_seen)
		losses = {}

		print("Predicting document {0}.".format(doc - 1))

		#### OUTDATED ####
		# With the help of some tensorflow-fu, we can now do a single forward pass per document
		# This results in a relative speedup of document prediction by 30%-50%
		with sess.as_default():
			generator = batch_generator(first_data, first_labels, BATCH_SIZE)

			loss_results = []
			for input_words, output_words in generator:
				feed_dict = {network_input: input_words, labels: output_words}
				loss_results += sess.run(o, feed_dict).tolist()

		for _pair in range(len(first_data)):
			if first_words[_pair] not in losses:
				losses[first_words[_pair]] = []
			losses[first_words[_pair]].append(loss_results[_pair])


		# Instantiate a POS-Tagger
		# If you choose not to do this NLTK will re-pickle
		# every time you tag a word, which takes ages.
		# This, again, might not be the most elegant way to do this
		# Return the modified frequency for a word or phrase
		def frequency(word):
			if len(word.split("_")) > 1:
				return min([frequency(i) for i in word.split("_")])
			else:
				if word in word_frequency:
					return word_frequency[word]
				else:
					return 1

		# Sort the set of words you encountered in the dictionary
		# by their average loss (in increasing order) and filter them
		# for stopwords
		s_l = sorted(losses.keys(), key=lambda x: -np.std(losses[x]))#*frequency(x))#sum(losses[x])/len(losses[x])*frequency(x))
		toPrint = []
		i = 0
		# while len(toPrint) < NUM_KEYWORDS or len(toPrint) == len(s_l):
		while i < len(s_l):
			s = s_l[i]
			i += 1
			if is_not_stopword(s, tagger):
				toPrint.append(str(s).replace("_", " "))

		i = 0
		while i < NUM_KEYWORDS and i < len(toPrint):
			if toPrint[i] in [word for wordsplit in toPrint for word in wordsplit.split(" ") if wordsplit != toPrint[i]]:
				toPrint = toPrint[:i] + toPrint[i+1:]

			# I dunno about this, it cleans up some definite noise, but I also think you might lose some good keywords
			# in the process.	
			elif toPrint[i].split("'")[0] in [word for word in toPrint if word != toPrint[i]]:
				toPrint = toPrint[:i] + toPrint[i+1:]
			i += 1

		toPrint = toPrint[:NUM_KEYWORDS]
		# Print the top 25 keywords
		# If --intersect is flagged, will only print if given the --verbose flag as well
		if (not INTERSECT) or VERBOSE:
			print(toPrint)

		# Remember the keywords this document generated
		keywords.append(toPrint)

		# I really guessed that printing the top 25 when sorting in
		# decreasing error would be more effective, but it seems like
		# the opposite is true. Regardless, for a sanity check, if you supply
		# the --difficult flag it will print out both alternatives.
		# Note the minus sign in the key argument.
		if SHOW_LEAST_PREDICTIVE:
			s_l = sorted(losses.keys(), key=lambda x: -sum(losses[x])/len(losses[x]))
			toPrint = []
			i = 0
			while i < len(s_l):
				s = s_l[i]
				i += 1
				if is_not_stopword(s, tagger):
					toPrint.append(str(s).replace("_", " "))
			# Print the top 25 keywords.
			print(toPrint)

	# This section triggers if you supply the --intersect flag, saying you want
	# a list of keywords representitive of the whole set of documents.
	if INTERSECT:

		# The percent of documents the keyword must appear in to be valid.
		# 25% is just a random guess.
		intersect_threshold = .25

		dict_of_words = {}

		# Wow, this is ugly. 
		# There must be a better way to do this, but maybe not.
		# Count the number of times each keyword appears as a keyword
		for keywordSet in keywords:
			for keyword in keywordSet:
				if keyword not in dict_of_words:
					dict_of_words[keyword] = 1
					for otherSet in keywords:
						if otherSet != keywordSet:
							if keyword in otherSet:
								dict_of_words[keyword] += 1

		# This determines our soft intersection of keywords.
		# Naively, a word is a keyword if it appears in more than intersect_threshhold document's keyword lists.
		# Additionally, I found that if a multi-word phrase appears as a keyword, it's likely to be a good keyword,
		# so all multi-word phrases are automatically included if they're found more than once.
		soft_intersection = [word for word in dict_of_words.keys() if dict_of_words[word] >= intersect_threshold * len(keywords) or len(word.split(" ")) > 1 and dict_of_words[word] > 2]
		print(soft_intersection)

	if PRINT or PRINT_PATH != None:

		with open(PRINT_PATH, "wba") as f:
			for doc_n, kwrds in enumerate(keywords):
				stri = "{0}: ".format(doc_n)
				for kwd_n, kwd in enumerate(kwrds):
					to_a = ""
					if kwd_n != 0:
						to_a += ", {0}".format(kwd.replace(" ", "_"))
					else:
						to_a += kwd
					stri += to_a

				f.write(stri + "\n")