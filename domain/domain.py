import fasttext
import numpy as np

from abc import abstractmethod
from typing import List

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics



class Utility:
    def __init__(self, X, Y):
        self.model = fasttext.load_model("CrawlMergeFasttextEmbeddingv3.bin")
        self.embeddedX = [self.model.get_sentence_vector(x) for x in X.tolist()]


        self.encoder = LabelEncoder()
        self.encoder.fit(Y)
        self.encodedY = self.encoder.transform(Y)
        self.label = self._getUnEncoddedY()

    def predict(self, texts: List[str], model):
        vectorizedTexts = [self.model.get_sentence_vector(text) for text in texts]
        results = model.predict(vectorizedTexts)
        return self.encoder.inverse_transform(results)

    def _getUnEncoddedY(self):
        uniqueGibberishLabel = np.unique(self.encodedY)
        return self.encoder.inverse_transform(uniqueGibberishLabel)

    def getScore(self, Y_pred, Y_test, average: str = "binary"):
        print("average: ",average)
        print("Accuracy: ", metrics.accuracy_score(Y_test, Y_pred))
        print("Precision: ", metrics.precision_score(Y_test, Y_pred, average=average))
        print("recall: ", metrics.recall_score(Y_test, Y_pred, average=average))
        print("F1-Score: ", metrics.f1_score(Y_test, Y_pred, average=average))
        print('\n============ Classification Report with {} ===============\n'.format(average))
        print(metrics.classification_report(Y_test, Y_pred, target_names=self.label))
        print('\n=============================================================\n')
