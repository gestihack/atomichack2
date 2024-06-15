import pdfplumber
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
import math
import pysbd
import re
import os


def pdf2txt(path):
    all_text = []
    all_pic = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            def filter_content(content):
                def filter_word(word):
                    if word['bottom'] > 64 and word['top'] < 762:
                        return True
                    else:
                        return False

                return list(filter(filter_word, content))

            filter_words = filter_content(page.extract_words())
            all_text.append(' '.join([t['text'] for t in filter_words])+f'------page{i}------')

            filter_lines = filter_content(page.extract_text_lines())
            for t in filter_lines:
                if t['text'].split(' ')[0].lower() == 'рисунок':
                    all_pic.append(t['text'])

        pdf.close()

    return ' '.join(all_text), all_pic


def rev_sigmoid(x:float)->float:
    return (1 / (1 + math.exp(0.5*x)))


def activate_similarities(similarities:np.array, p_size=10)->np.array:
        x = np.linspace(-10,10,p_size)
        y = np.vectorize(rev_sigmoid)

        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        diagonals = np.stack(diagonals)
        diagonals = diagonals * activation_weights.reshape(-1,1)
        activated_similarities = np.sum(diagonals, axis=0)

        return activated_similarities


def split_content(sentences):
    sentece_length = [len(each) for each in sentences]
    long = np.mean(sentece_length) + np.std(sentece_length) *2
    short = np.mean(sentece_length) - np.std(sentece_length) *2

    text = ''
    for each in sentences:
        if len(each) > long:
            comma_splitted = each.replace(',', '.')
        else:
            text+= f'{each}. '
    sentences = text.split('. ')

    text = ''
    for each in sentences:
        if len(each) < short:
            text+= f'{each} '
        else:
            text+= f'{each}. '

    sentences = text.split('. ')

    embeddings = model.encode(sentences)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    similarities = cosine_similarity(embeddings)

    activated_similarities = activate_similarities(similarities, p_size=10)
    minmimas = argrelextrema(activated_similarities, np.less, order=2)

    split_points = [each for each in minmimas[0]]
    text = ''
    for num,each in enumerate(sentences):
        if num in split_points:
            text+=f'\n\n {each} '
        else:
            text+=f'{each} '

    return text


seg = pysbd.Segmenter(language="ru", clean=True)
model = SentenceTransformer('all-mpnet-base-v2')
path = "C:\\Users\sevco\Рабочий стол\Для Хакатона\\"
out_path = "C:\\Users\sevco\Рабочий стол\excel\\"

print("Files and directories in '", path, "' :")


dir_list = os.listdir(path)
print(dir_list)

for filename in dir_list:
    print(filename)
    file = path + filename
    out_file = out_path+filename+'.xlsx'
    txt, pic = pdf2txt(file)
    sentence = seg.segment(txt)

    res = split_content(sentence)
    for elem in pic:
        res = res.replace(elem, '')

    num = [''.join(re.findall(r'\d', elem)) for elem in re.findall(r'[-]{6}page\d+[-]{6}', res)]
    pages = re.split(r'[-]{6}page\d+[-]{6}', res)
    dic = list(zip(pages, num))

    print(dic)

    for elem in dic:
        if '\n\n' in elem[0]:
            for i in elem[0].split('\n\n'):
                dic.append([i, elem[1]])
            dic.remove(elem)

    frame = pd.DataFrame(dic)
    frame.to_excel(out_file, index=False)
    print(f'файл {out_file} успешно создан')

