
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load components
tokenizer = joblib.load("tokenizer.joblib")
label_encoder = joblib.load("label_encoder.joblib")
model = load_model("ann_language_model.h5")

# Prediction loop
max_len = 200
print("ANN Language Detection (type 'exit' to quit)")
while True:
    text = input("Enter text: ")
    if text.lower() == 'exit':
        break
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    lang = label_encoder.inverse_transform([np.argmax(pred)])
    print("Detected Language:", lang[0])
