import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from sentence_transformers import SentenceTransformer, util
import numpy as np
import csv 
import pickle
import json
import os

qtcreator_file  = "qt.ui" # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.find_Button.clicked.connect(self.find)





#for name, param in embedder.named_parameters():
#    if param.requires_grad:print(name)
    def find(self):
        embedder = SentenceTransformer('model')
        with open('reverse.json', 'r') as f:
            reverse = json.load(f)

# Corpus with example sentences
        with open('pQcollection.csv', newline = '', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)
        corpus = data[0]

        if not os.path.isfile("embed.pickle"):
            with open("embed.pickle",'wb') as fil:
                corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
                pickle.dump(corpus_embeddings, fil)
        else:
            with open("embed.pickle",'rb') as fil:
                corpus_embeddings = pickle.load(fil)
                print('loaded')
            
        #corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
        #os.path.isfile('')
        # Query sentences:

        queries = self.input_textEdit.toPlainText()
        queries = [queries ]

        out_tag = []

        # Find the closest 6 sentences of the corpus for each query sentence based on cosine similarity
        top_k = 6
        for query in queries:
            query_embedding = embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()

            #We use np.argpartition, to only partially sort the top_k results
            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

            out = [] 
            
            tags ={}
            for idx in top_results[0:top_k]:
                if reverse[corpus[idx]][2] != []:
                    for tag in reverse[corpus[idx]][2]:
                        if tag in tags:
                            tags[tag] +=1
                        else:
                            tags[tag] = 1
                out.append("https://leetcode.com/problems/"+ reverse[corpus[idx]][0])
                out.append(corpus[idx].strip())
                out.append( "(Score: %.4f)" % (cos_scores[idx]))
                out.append("===========================")
                
            
            for key, value in sorted(tags.items(), key=lambda item: item[1], reverse=True):
                out_tag.append("%s: %s" % (key, value))
                
            self.out_listWidget.clear()
            self.out_listWidget.addItems(out_tag)
            self.listWidget.clear()
            self.listWidget.addItems(out)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())