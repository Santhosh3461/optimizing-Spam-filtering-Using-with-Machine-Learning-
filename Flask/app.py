from flask import Flask, render_template, request
import pickle 
import numpy as np
import re
# import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from keras.models import load_model
import os
print("Pavin")
app = Flask(__name__)

# load the trained model and other required files
loaded_model = load_model('spam.h5')
cv = pickle.load(open('ck1.pkl', 'rb'))

# define the home route
@app.route('/')
def home():
    return render_template('index.html')

# define the route for the form
@app.route('/Spam', methods=['GET', 'POST'])
def spam():
    if request.method == 'POST':
        message = request.form['message']
        new_review = str(message)
        new_review = re.sub('[^a-zA-Z]', ' ', new_review)
        new_review = new_review.lower()
        new_review = new_review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
        new_review = ' '.join(new_review)
        new_corpus = [new_review]
        new_X_test = cv.transform(new_corpus).toarray()
        new_y_pred = loaded_model.predict(new_X_test)
        new_X_pred = np.where(new_y_pred > 0.5, 1, 0)
        if new_X_pred[0][0] == 1:
            return render_template('result.html', prediction='Spam')
        else:
            return render_template('result.html', prediction='Not a Spam')
    else:
        return render_template('spam.html')

# run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False)
