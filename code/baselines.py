from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import scale
from sklearn.metrics import f1_score
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import NMF
import pickle
import os

DATASET_PATH = "../datasets/bt.1.0/"

OTHER_PATH = "../datasets/SemEval2010/"

OUTFILE = "../datasets/bt.1.0/bitterlemons_files.p"
OUTFILE_STANCE = "../datasets/bt.1.0/bitterlemons_stance.p"

SEM_OUTFILE = OTHER_PATH + "train.data.p"



def parseToPickle():
	X = []
	y = []
	for filename in os.listdir(OTHER_PATH + "train/"):
		docname = OTHER_PATH + "train/" + filename
		f = open(docname, 'r').read()
		X.append(f)
	f = open(SEM_OUTFILE, "wb")
	pickle.dump([X], f)
	f.close()


def parseToBOW():
	vectorizer = CountVectorizer(min_df=1)
	texts = pickle.load(open(OUTFILE, 'rb'))[0]
	tdm = vectorizer.fit_transform(texts)
	transformer = TfidfTransformer()
	tdidf = transformer.fit_transform(tdm)
	f = open(DATASET_PATH + "BOW.p", "wb")
	pickle.dump(tdm, f)
	f.close()
	f = open(DATASET_PATH + "BOW_TDIDF.p", "wb")
	pickle.dump(tdidf, f)
	f.close()

def predict_accuracy(true_labels, predictions):
	numerical_mapped_1 = [0 if i == "Israeli" else 1 for i in true_labels]
	numerical_mapped_2 = [1 if i == "Israeli" else 0 for i in true_labels]
	one = f1_score(numerical_mapped_1, predictions, average='weighted')
	two = f1_score(numerical_mapped_2, predictions, average='weighted')
	return str(max(one, two))

def KMeansAccuracy():
	clusterer = KMeans(n_clusters=2, n_init=30)
	tdm = pickle.load(open(DATASET_PATH + "BOW.p", "rb"))
	predictions = clusterer.fit_predict(tdm)
	true_labels = pickle.load(open(OUTFILE_STANCE, "rb"))[0]
	numerical_mapped_1 = [0 if i == "Israeli" else 1 for i in true_labels]
	numerical_mapped_2 = [1 if i == "Israeli" else 0 for i in true_labels]
	one = f1_score(numerical_mapped_1, predictions)
	two = f1_score(numerical_mapped_2, predictions)
	print("The F1 score of KMeans on BOW is: " + str(max(one, two)))

	clusterer = KMeans(n_clusters=2, n_init=30)
	predictions = clusterer.fit_predict(tdm)
	true_labels = pickle.load(open(OUTFILE_STANCE, "rb"))[0]
	accuracy = predict_accuracy(true_labels, predictions)
	print("The F1 score of KMeans on BOW (w/Tdidf) is: " + accuracy)


def cluster_kmeans(matrix, true_labels):
	clusterer = KMeans(n_clusters=2, n_init=30)
	predictions = clusterer.fit_predict(matrix)
	accuracy = predict_accuracy(true_labels, predictions)
	print("The F1 score is: " + accuracy)


def nmf_accuracy():
	tdm = pickle.load(open(DATASET_PATH + "BOW.p", "rb"))
	true_labels = pickle.load(open(OUTFILE_STANCE, "rb"))[0]
	print("I'm NNMF-ing!")
	NNMF = NMF(max_iter=50, n_components=100)
	tdm_reshaped = NNMF.fit_transform(tdm)
	print("I'm clustering!")
	cluster_kmeans(tdm_reshaped, true_labels)

def SpectralAccuracy():
	clusterer = SpectralClustering(n_clusters=2)
	tdm = pickle.load(open(DATASET_PATH + "BOW_TDIDF.p", "rb"))
	predictions = clusterer.fit_predict(tdm)
	true_labels = pickle.load(open(OUTFILE_STANCE, "rb"))[0]
	numerical_mapped_1 = [0 if i == "Israeli" else 1 for i in true_labels]
	numerical_mapped_2 = [1 if i == "Israeli" else 0 for i in true_labels]
	one = f1_score(numerical_mapped_1, predictions)
	two = f1_score(numerical_mapped_2, predictions)
	print("The F1 score of Spectral Clustering on BOW (w/Tdidf) is: " + str(max(one, two)))

parseToPickle()