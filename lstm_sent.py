import pandas as pd
import io
import json
import nltk
import re
from numpy import asarray, zeros
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout

nltk.download('stopwords')

#load the training data
data_train = pd.read_csv('train_data.csv')
train_data = data_train['review']
train_label = data_train['sentiment']

#load the testing data
data_test = pd.read_csv('test_data.csv')
test_data = data_test['text']

print("test data: ", test_data)

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    #removes HTML tags
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    
    sentence = sen.lower()

    #remove html tags
    sentence = remove_tags(sentence)

    #remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    #single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  

    #remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence) 

    #remove stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

train = []
reviews = list(train_data)
for review in reviews: 
    train.append(preprocess_text(review))

train_data = train

#combine the text data from both datasets for preprocessing
data = pd.concat([pd.Series(train_data), test_data])

#perform label encoding on the sentiment labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_label)

#tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

tokenizer_json = tokenizer.to_json()
with io.open('b3_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

#pad sequences
vocab_size = len(tokenizer.word_index) + 1

max_length = 150

train_data = pad_sequences(train_sequences, padding='post', maxlen=max_length)
test_data = pad_sequences(test_sequences, padding='post', maxlen=max_length)

#load GloVe word embeddings and create an Embeddings Dictionary
embeddings_dictionary = dict()
glove_file = open('glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()
 
#create Embedding Matrix having 100 columns 
#containing 100-dimensional GloVe word embeddings for all words in our corpus.
embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# embedding_matrix.shape

#define the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 100, weights = [embedding_matrix], input_length=max_length))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#compile and train model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_data, train_labels, epochs=4, batch_size=32)

#predict sentiment on the test data
predictions = model.predict(test_data)
print("predictions: ", predictions)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]

#decode the predicted sentiment labels
predicted_sentiment = label_encoder.inverse_transform(predicted_classes)

print(predicted_sentiment)
