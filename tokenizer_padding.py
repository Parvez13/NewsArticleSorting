from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

# Add tokens to our words and padding to the tokens according to sentence length.
def tokenizer_and_padding(X_train_sent, X_test_sent, max_features=10000):
  max_features=max_features
  # tokenizer
  tokenizer = Tokenizer(num_words=max_features)
  tokenizer.fit_on_texts(X_train_sent)
  # texts_to_sequences
  X_train = tokenizer.texts_to_sequences(X_train_sent)
  X_test = tokenizer.texts_to_sequences(X_test_sent)
  # pad_sequences
  output_seq_len = 593 # 90% of the sentences having length
  X_train = pad_sequences(X_train, padding = 'post',maxlen=output_seq_len)
  X_test  =  pad_sequences(X_test, padding = 'post',maxlen=output_seq_len)
  return X_train, X_test, tokenizer
