# Hyper parameters
word_embedding_size = 100 # word embedding size
char_embedding_size = 100 # char embedding size
kernels = [2, 3, 4, 5] # CNN filter sizes, use window 3, 4, 5 for char CNN
char_hidden_size = 200 # CNN output
lstm_hidden_size = 300 # LSTM hidden size

converge_check = 30
use_chars = True
char_embedding_method = "hcnn"
context_window = 5
#char_embedding_method = "lstm"
#char_embedding_method = "vcnn"
use_crf = True
use_char_attention = True
clip = 5
batch_size = 20
num_epochs = 35
dropout = 0.5
learning_rate = 0.001
learning_rate_decay = 0.9
