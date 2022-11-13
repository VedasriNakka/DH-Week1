

import os
import numpy as np
import matplotlib.pyplot as pltcomedy
import nltk
import nltk.tokenize
import pandas as pd
import re
import collections
import lxml.etree
import tarfile
import math


import numpy as np
import matplotlib.pyplot as plt

document_term_matrix = np.array([[1, 16], [4, 18], [35, 2], [10, 3]])
labels = '$d_1$', '$d_2$', '$d_3$', '$Q$'
plt.ylim(0, 20); plt.xlim(0, 44)
plt.quiver([0, 0, 0, 0], [0, 0, 0, 0], document_term_matrix[:, 0], document_term_matrix[:, 1],
    color=["C0", "C0", "C0", "C2"], angles='xy', scale_units='xy', scale=1)
for i, label in enumerate(labels):
    plt.annotate(label, xy=document_term_matrix[i], fontsize=15)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2");
plt.show()




PUNCT_RE = re.compile(r'[^\w\s]+$')


def is_punct(string):
    """Check if STRING is a punctuation marker or a sequence of
       punctuation markers.
    """
    return PUNCT_RE.match(string) is not None

def preprocess_text(text, language='French', lowercase=True):
    if lowercase:
        text = text.lower()
    if (language == 'French'):
        text = re.sub("-", " ", text)
        text = re.sub("l'", "le ", text)
        text = re.sub("d'", "de ", text)
        text = re.sub("c'", "ce ", text)
        text = re.sub("j'", "je ", text)
        text = re.sub("m'", "me ", text)
        text = re.sub("qu'", "que ", text)
        text = re.sub("'", " ' ", text)
        text = re.sub("quelqu'", "quelque ", text)
        text = re.sub("aujourd'hui", "aujourdhui", text)
    tokens = nltk.tokenize.word_tokenize(text, language=language)
    tokens = [token for token in tokens if not is_punct(token)]
    return tokens

# Start the job

subgenres = ('Comédie', 'Tragédie', 'Tragi-comédie')
plays, titles, genres = [], [], []
authors, years = [], []

for fn in os.scandir('data/theatre-classique'):
    # Only include XML files
    if not fn.name.endswith('.xml'):
        continue
    tree   = lxml.etree.parse(fn.path)
    genre  = tree.find('//genre')
    title  = tree.find('//title')
    author = tree.find('//author')
    year   = tree.find('//date')
    if genre is not None and genre.text in subgenres:
        lines = []
        for line in tree.xpath('//l|//p'):
            lines.append(' '.join(line.itertext()))
        text = '\n'.join(lines)
        plays.append(text)
        genres.append(genre.text)
        titles.append(title.text)
        authors.append(author.text)
        if year is not None:
            years.append(year.text)

print (len(plays), len(genres), len(titles), len(authors), len(years))
#
# Overview Statistics
counts = collections.Counter(genres)

fig, ax = plt.subplots()
ax.bar(counts.keys(), counts.values(), width=0.3)
ax.set(xlabel="genre", ylabel="count");
fig.show()

# %% tokenization takes time
plays_tok = [preprocess_text(play, 'French') for play in plays]

# %%
def extract_vocabulary(tokenized_corpus, min_count=1, max_count=float('inf')):
    vocabulary = collections.Counter()
    for document in tokenized_corpus:
        vocabulary.update(document)
    vocabulary = {word for word, count in vocabulary.items()
                  if count >= min_count and count <= max_count}
    return sorted(vocabulary)

vocabulary = extract_vocabulary(plays_tok, min_count=2)
len(vocabulary)

# rtf instead of tf
dd = collections.Counter(docs_tok[0])
vocSize = len(voc_docs)
epsilon = 1.0 / vocSize
document_term_matrix = []
row = [dd[word] for word in voc_docs]
aSum = sum(row)
for anIndex in range(vocSize):
    aValue = row[anIndex]
    if (aValue > 0):
        aValue = (aValue+1.0) / aSum
    else:
        aValue = epsilon
    row[anIndex] = aValue
    
document_term_matrix.append(row)


# %%  very very slow!!
bags_of_words = []
for document in plays_tok:
    tokens = [word for word in document if word in vocabulary]
    bags_of_words.append(collections.Counter(tokens))

print(bags_of_words[2])

#
# Example of a doc x term matrix
test_corpus = ["a aa aaa a a a a aa aa aa",
          "b bb bbb bb bb bb  bb bb a aa aaa",
          "c cc ccc cc cc cc c c c ccc aaa aaa aaa aaa",
          "d dd ddd ddd ddd ddd ddd d dd aa bb bb aa cc cc "]

# Generate the document vectors (tokenization)
docs_tok=[]
for anIndex in range(len(test_corpus)):
   docs_tok.append( preprocess_text(test_corpus[anIndex], 'French') )

# the associated vocabulary
voc_docs = extract_vocabulary(docs_tok, min_count=1)
voc_docs

#
# Representation in a doc x term matrix for the French plays
def corpus2dtm(tokenized_corpus, vocabulary):
    "Transform a tokenized corpus into a document-term matrix"
    document_term_matrix = []
    for document in tokenized_corpus:
        document_counts = collections.Counter(document)
        row = [document_counts[word] for word in vocabulary]
        document_term_matrix.append(row)
    return document_term_matrix

# builing the doc/term matrix in a few seconds for the French example
document_term_matrix = np.array(corpus2dtm(plays_tok, vocabulary))
print(f"document-term matrix with "
      f"|D| = {document_term_matrix.shape[0]} documents and "
      f"|V| = {document_term_matrix.shape[1]} words.")

# Builing the doc/term matrix for the toy-size example
dtm  = np.array(corpus2dtm(docs_tok, voc_docs))
print(f"document-term matrix with "
      f"|D| = {dtm.shape[0]} documents and "
      f"|V| = {dtm.shape[1]} words.")

 # examples of manipulation of the doc x term matrix
dtm
dtm[0]   # a single doc
voc_docs.index('a')
dtm[0, 0:4]
dtm[:, 0:3]  # the first 3 terms for all docs
anIndex = voc_docs.index('a')
dtm[:, anIndex]   # a single term over all docs

for anIndex in range(len(voc_docs)):
   print (anIndex, dtm[:, anIndex])

for anIndex in range(dtm.shape[0]):
   print (anIndex, dtm[anIndex, :])


#
# If you want the rtf values instead of the tf values
def corpus2dtmRelative(tokenized_corpus, vocabulary, aLambda=1.0):
    "Transform a tokenized corpus into a document-term matrix with relative term frequency."
    document_term_matrix = []
    vocSize = len(vocabulary)
    for document in tokenized_corpus:
        document_counts = collections.Counter(document)
        row = [document_counts[word] for word in vocabulary]
        aSum = sum(row) + (vocSize*aLambda)
        for anIndex in range(vocSize):
            aValue = row[anIndex]
            if (aValue > 0):
                aValue = (aValue+aLambda) / aSum
            else:
                aValue = aLambda / aSum
            row[anIndex] = aValue
        document_term_matrix.append(row)
    return document_term_matrix

# builing the doc/term matrix for the toy-size example
dtmR = np.array(corpus2dtmRelative(docs_tok, voc_docs))
print(f"document-term matrix with "
      f"|D| = {dtmR.shape[0]} documents and "
      f"|V| = {dtmR.shape[1]} words.")


nbDoc = dtm.shape[0]
aDoc = dtm[0]
for anIndex in range(nbDoc):
    aColumn = dtm[anIndex]
    d1 = euclidean_distance(aDoc, aColumn)
    d2 = cosine_distance(aDoc, aColumn)
    print(f'{anIndex:}  {d1:.3f}  {d2:.3f}  {d3:.3f}')

    d2 = city_block_distance(aDoc, aColumn)
    d3 = cosine_distance(aDoc, aColumn)


# %%
monsieur_idx = vocabulary.index('monsieur')
sang_idx = vocabulary.index('sang')
monsieur_counts = document_term_matrix[:, monsieur_idx]
sang_counts = document_term_matrix[:, sang_idx]

# %%
genres = np.array(genres)

# %%
fig, ax = plt.subplots()

for genre in ('Comédie', 'Tragédie', 'Tragi-comédie'):
    ax.scatter(monsieur_counts[genres == genre],
               sang_counts[genres == genre],
               label=genre, alpha=0.7)

ax.set(xlabel='monsieur', ylabel='sang')
plt.legend();
plt.show()

# %%
tr_means = document_term_matrix[genres == 'Tragédie'].mean(axis=0)
co_means = document_term_matrix[genres == 'Comédie'].mean(axis=0)
tc_means = document_term_matrix[genres == 'Tragi-comédie'].mean(axis=0)

# %%
print( len(tr_means) )

# %%
print('Mean absolute frequency of "monsieur"')
print(f'   in comédies: {co_means[monsieur_idx]:.2f}')
print(f'   in tragédies: {tr_means[monsieur_idx]:.2f}')
print(f'   in tragi-comédies: {tc_means[monsieur_idx]:.2f}')


# %%
fig, ax = plt.subplots()

ax.scatter(
    co_means[monsieur_idx], co_means[sang_idx], label='Comédies')
ax.scatter(
    tr_means[monsieur_idx], tr_means[sang_idx], label='Tragédie')
ax.scatter(
    tc_means[monsieur_idx], tc_means[sang_idx], label='Tragi-comédies')

ax.set(xlabel='monsieur', ylabel='sang')
plt.legend();
plt.show()

# %%
tragedy = np.array([tr_means[monsieur_idx], tr_means[sang_idx]])
comedy = np.array([co_means[monsieur_idx], co_means[sang_idx]])
tragedy_comedy = np.array([tc_means[monsieur_idx], tc_means[sang_idx]])


# %%
# show the three points for the three text genres
fig, ax = plt.subplots()
ax.scatter(co_means[monsieur_idx], co_means[sang_idx],
           label='Comédies', zorder=3)
ax.scatter(tr_means[monsieur_idx], tr_means[sang_idx],
           label='Tragédie', zorder=3)
ax.scatter(tc_means[monsieur_idx], tc_means[sang_idx],
           label='Tragi-comédies', zorder=3)
ax.set(xlabel='monsieur', ylabel='sang')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3);
plt.show()


# %%
# Same plot but the the lines (according to Euclidan distance)
fig, ax = plt.subplots()
ax.plot([tr_means[monsieur_idx], tc_means[monsieur_idx]],
        [tr_means[sang_idx], tc_means[sang_idx]],
        'darkgrey', lw=2, ls='--')
ax.plot([tr_means[monsieur_idx], co_means[monsieur_idx]],
        [tr_means[sang_idx], co_means[sang_idx]],
        'darkgrey', lw=2, ls='--')
ax.plot([tc_means[monsieur_idx], co_means[monsieur_idx]],
        [tc_means[sang_idx], co_means[sang_idx]],
        'darkgrey', lw=2, ls='--')
ax.scatter(co_means[monsieur_idx], co_means[sang_idx],
           label='Comédies', zorder=3)
ax.scatter(tr_means[monsieur_idx], tr_means[sang_idx],
           label='Tragédie', zorder=3)
ax.scatter(tc_means[monsieur_idx], tc_means[sang_idx],
           label='Tragi-comédies', zorder=3)
ax.set(xlabel='monsieur', ylabel='sang')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3);
plt.show()


# 
# Compute the Euclidan distance

# %%
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# %%
tc = euclidean_distance(tragedy, comedy)
print(f'tragédies - comédies:       {tc:.2f}')
ttc = euclidean_distance(tragedy, tragedy_comedy)
print(f'tragédies - tragi-comédies: {ttc:.2f}')
ctc = euclidean_distance(comedy, tragedy_comedy)
print(f' comédies - tragi-comédies: {ctc:.2f}')

# Manhattan distance
def city_block_distance(a, b):
    return ( np.abs(a - b).sum() )

tc = city_block_distance(tragedy, comedy)
ttc = city_block_distance(tragedy, tragedy_comedy)
ctc = city_block_distance(comedy, tragedy_comedy)
print(f'tragédies - comédies:       {tc:.2f}')
print(f'tragédies - tragi-comédies: {ttc:.2f}')
print(f' comédies - tragi-comédies: {ctc:.2f}')

# dot product
def dot_product_distance(a, b):
    return (np.dot(a, b))

tc = dot_product_distance(tragedy, comedy)
ttc = dot_product_distance(tragedy, tragedy_comedy)
ctc = dot_product_distance(comedy, tragedy_comedy)
print(f'tragédies - comédies:       {tc:.2f}')
print(f'tragédies - tragi-comédies: {ttc:.2f}')
print(f' comédies - tragi-comédies: {ctc:.2f}')


# Vector norm or vector length
def vector_len(v):
    """Compute the length (or norm) of a vector."""
    return (np.sqrt(np.sum(v ** 2)) )

# Cosine distance
def cosine_distance(a, b):
    return ( 1 - np.dot(a, b) / (vector_len(a) * vector_len(b)) )

tc = cosine_distance(tragedy, comedy)
ttc = cosine_distance(tragedy, tragedy_comedy)
ctc = cosine_distance(comedy, tragedy_comedy)
print(f'tragédies - comédies:       {tc:.2f}')
print(f'tragédies - tragi-comédies: {ttc:.2f}')
print(f' comédies - tragi-comédies: {ctc:.2f}')




