
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load data
df = pd.read_csv("Language Detection.csv")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Language"])
joblib.dump(label_encoder, "label_encoder.joblib")

# Tokenize text (character-level)
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df["Text"])
X_seq = tokenizer.texts_to_sequences(df["Text"])
joblib.dump(tokenizer, "tokenizer.joblib")

# Pad sequences
max_len = 200
X_pad = pad_sequences(X_seq, maxlen=max_len)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Build ANN model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile and train
model.compile(optimizer='omar', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Save model
model.save("ann_language_model.h5")
