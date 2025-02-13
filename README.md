Overview:
This project trains an LSTM model for sentiment analysis using the Sentiment140 dataset. It processes tweets to classify them as positive or negative.

Dataset:
Sentiment140 dataset with 1.6 million labeled tweets are fetched from kaggle.
Labels:
0 = Negative
4 (converted to 1) = Positive

Preprocessing:
Removed unnecessary columns and null values.
Clean text: lowercase, removed non-alphabetic characters.
Tokenization, removal of stopwords, and applied stemming.
Converted text to sequences and pad to max length of 50.

Model Architecture:
Embedding Layer for word representation.
LSTM Layers with dropout to prevent overfitting.
Dense Layer with sigmoid activation for classification.
Compiled with Adam optimizer and binary crossentropy loss.

Training Details:
Batch size: 64
Epochs: 5
20% data used for validation.
