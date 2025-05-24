from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import numpy as np
from underthesea import word_tokenize

# Flask app
app = Flask(__name__)

# Load mô hình khi khởi động server
svm_model = joblib.load('Model v2.pkl')  # Mô hình SVM
# lstm_model = tf.keras.models.load_model('lstm.keras')  # Mô hình LSTM
# multi_model = tf.keras.models.load_model('multichain.keras')  # Mô hình Multi-channel LSTM-CNN

# Load vectorizer/tokenizer tương ứng
vectorizer = joblib.load('TFIDF Vectorizer v2.pkl')

MAX_LEN = 256

def predict_sentiment_svm(sentence):
    processed = word_tokenize(sentence, format="text")
    tfidf_vector = vectorizer.transform([processed])
    prediction = svm_model.predict(tfidf_vector)[0]
    prob_all = svm_model.predict_proba(tfidf_vector)[0]
    return prediction, prob_all

reverse_label_map = {0: 'Tích cực', 1: 'Trung lập', 2: 'Tiêu cực'}

@app.route('/api/sentiment')
def sentiment():
    text = request.args.get('text')
    model_id = request.args.get('model')

    if not text or not model_id:
        return jsonify({'error': 'Missing text or model parameter'}), 400

    try:
        model_id = int(model_id)
    except ValueError:
        return jsonify({'error': 'model must be an integer'}), 400

    if model_id == 1:
        # Dự đoán bằng SVM
        pred, prob = predict_sentiment_svm(text)
        return {
            "prediction": int(pred),
            "probabilities": prob.tolist()
        }


    # elif model_id == 2:
    #     # Dự đoán bằng LSTM
    #     seq = tokenizer.texts_to_sequences([text])
    #     padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    #     prediction = np.argmax(lstm_model.predict(padded), axis=1)[0]
    #
    # elif model_id == 3:
    #     # Dự đoán bằng Multi-channel LSTM-CNN
    #     seq = tokenizer.texts_to_sequences([text])
    #     padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    #     prediction = np.argmax(multi_model.predict(padded), axis=1)[0]

    else:
        return jsonify({'error': 'Invalid model ID'}), 400

    return jsonify({
        'text': text,
        'model': model_id,
        'prediction': int(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
