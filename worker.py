import re
import string
import zipfile
import os.path
import numpy as np
import onnxruntime as rt
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

if not os.path.isfile('models/log_reg.onnx'):
    with zipfile.ZipFile('models/log_reg.zip', 'r') as zip_ref:
        zip_ref.extractall('models')

sess = rt.InferenceSession("models/log_reg.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

def preprocess(text:str):
    text = re.sub('[?!]+',' featuremark ',text)
    text = re.sub('\.{2,}',' featuredot ',text)
    text = re.sub('[0-9]+','',text)

    words = TweetTokenizer().tokenize(text)
    punct = list(string.punctuation)

    words = [word.lower() if not word.isupper() else word for word in words ]
    custom_sw = ["'s","``","'m","'d","'re","--", "(",")","'d",""," ","n't","'t","'"]
    sw = set(list(stopwords.words('english')) + punct + custom_sw)
    words = [word for word in words if word not in sw]
    preprocessed_text = ' '.join(words)

    return preprocessed_text

def detect(text):
    input = np.array(preprocess(text)).reshape(-1,1)
    result = sess.run([output_name], {input_name: input})
    return result[0]
