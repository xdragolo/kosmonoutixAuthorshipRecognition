import numpy as np
import pandas
import seaborn as sn
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from dataPreprocessing import cleanData, vectorizeArticles
from model import getLSTM
from tensorflow import keras
from sklearn.model_selection import GridSearchCV

# setting
n_words = 80000
seq_length = 600
nAuthors = 5
epochs = 6
embedding_dim =150


# data cleaning and and analysis
df = pandas.read_json('./Scraper/kosmonautix.json')
df, nAuthors = cleanData(df)


# data vectorizing
df = pandas.read_csv('./cleanData.csv', sep=';')
df = df[['author', 'content']]
X, Y = vectorizeArticles(df, n_words=n_words, seq_length=seq_length)
categories = Y.columns.tolist()
Y = Y.values

# train 60% test 20% validation 20%, validation cut is made during fitting
# Y in different format for GridSearch
# authorsDict = {}
# for i in range(nAuthors):
#     authorsDict[categories[i]] = i
# df = df.replace({'author' : authorsDict})
# Y = df['author'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# parameters setting
# param_grid = {
#     'embedding_dim' : [50,100,150,200],
# }
# classifier = KerasClassifier(build_fn=getLSTM)
# grid = GridSearchCV(estimator=classifier, param_grid=param_grid, n_jobs=-1, return_train_score=True,scoring='accuracy')
# grid_result = grid.fit(X_train, y_train)
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

#  model fitting
model = getLSTM(n_words, input_length=X.shape[1], embedding_dim=embedding_dim, n_outputs=nAuthors)
# print(model.summary())
history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
model.save('./models/')

# learning curves
# print(history.history.keys())
# plt.figure(figsize=(10,3))
# plt.subplot(111)
# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show();
#
# plt.subplot(122)
# plt.title('Accuracy')
# plt.plot(history.history['accuracy'], label='train')
# plt.plot(history.history['val_accuracy'], label='test')
# plt.legend()
# plt.show();


# model evaluation
model = keras.models.load_model('./models/')
accr = model.evaluate(X_test, y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

y_pred = model.predict(X_test)
y_pred_labeled = []
for idx in y_pred.argmax(axis=1): y_pred_labeled.append(categories[idx])
y_test_labeled = []
for idx in y_test.argmax(axis=1): y_test_labeled.append(categories[idx])

# print(y_pred_labeled)
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1) )
sn.heatmap(matrix, xticklabels=categories, yticklabels=categories)
plt.savefig('./figures/confusionMatrix.png', bbox_inches='tight')
plt.show()