"""
Humanities Data Analysis: Case studies with Python
--------------------------------------------------
Folgert Karsdorp, Mike Kestemont & Allen Riddell
Chapter 2: Parsing and Manipulating Structured Data
"""


import os

# %%
import tarfile
tf = tarfile.open('data/folger.tar.gz', 'r')
tf.extractall('data')


# %%
file_path = 'data/folger/txt/1H4.txt'
stream = open(file_path)
contents = stream.read()
stream.close()

print(contents[:300])


# %%
with open(file_path) as stream:
    contents = stream.read()

print(contents[:300])


# Moby Dick

file_path = 'data/MobyDick.txt'	# a second example
with open(file_path) as stream:	# read the entire file
    contents = stream.read()

print(contents[:196])


# %%
with open('data/anna-karenina.txt', encoding='koi8-r') as stream:
    # Use stream.readline() to retrieve the next line from a file,
    # in this case the 1st one:
    line = stream.readline()

print(line)

  
# %%
csv_file = 'data/folger_shakespeare_collection.csv'
with open(csv_file) as stream:
    # call stream.readlines() to read all lines in the CSV file as a list.
    lines = stream.readlines()

print(lines[:3])


# %%
entries = []
for line in open(csv_file):
    entries.append(line.strip().split(','))

for entry in entries[:3]:
    print(entry)


# %%
import csv

entries = []
with open(csv_file) as stream:
    reader = csv.reader(stream, delimiter=',')
    for fname, author, title, editor, publisher, pubplace, date in reader:
        entries.append((fname, title))

for entry in entries[:5]:
    print(entry)


# %%
entries = []
with open(csv_file) as stream:
    reader = csv.reader(stream, delimiter=',')
    for fname, _, title, *_ in reader:
        entries.append((fname, title))

for entry in entries[:5]:
    print(entry)


# %%
entries = []

with open(csv_file) as stream:
    reader = csv.DictReader(stream, delimiter=',')
    for row in reader:
        entries.append(row)

for entry in entries[:5]:
    print(entry['fname'], entry['title'])


# %%
import PyPDF2 as PDF


# %%
file_path = 'data/folger/pdf/1H4.pdf'
pdf = PDF.PdfFileReader(file_path, overwriteWarnings=False)


# %%
n_pages = pdf.getNumPages()
print(f'PDF has {n_pages} pages.')


# %%
page = pdf.getPage(1)
content = page.extractText()
print(content[:150])


# %%
def pdf2txt(fname, page_numbers=None, concatenate=False):
    """Convert text from a PDF file into a string or list of strings.

    Arguments:
        fname: a string pointing to the filename of the PDF file
        page_numbers: an integer or sequence of integers pointing to the
            pages to extract. If None (default), all pages are extracted.
        concatenate: a boolean indicating whether to concatenate the
            extracted pages into a single string. When False, a list of
            strings is returned.

    Returns:
        A string or list of strings representing the text extracted
        from the supplied PDF file.

    """
    pdf = PDF.PdfFileReader(fname, overwriteWarnings=False)
    if page_numbers is None:
        page_numbers = range(pdf.getNumPages())
    elif isinstance(page_numbers, int):
        page_numbers = [page_numbers]
    texts = [pdf.getPage(n).extractText() for n in page_numbers]
    return '\n'.join(texts) if concatenate else texts


# %%
text = pdf2txt(file_path, concatenate=True)
sample = pdf2txt(file_path, page_numbers=[1, 4, 9])


# %%
import json

line = {
    'line_id': 12664,
    'play_name': 'Alls well that ends well',
    'speech_number': 1,
    'line_number': '1.1.1',
    'speaker': 'COUNTESS',
    'text_entry': 'In delivering my son from me, I bury a second husband.'
}

print(json.dumps(line))


# %%
with open('shakespeare.json', 'w') as f:
    json.dump(line, f)


# %%
with open('data/macbeth.json') as f:
    data = json.load(f)

print(data[3:5])


# %%
import collections

languages = collections.Counter()
for entry in data:
    languages[entry['lang']] += 1

print(languages.most_common())





#
#  XML format
#
#

# %%
with open('data/sonnets/18.xml') as stream:
    xml = stream.read()

print(xml)


# %%
import lxml.etree


# %%
tree = lxml.etree.parse('data/sonnets/18.xml')
print(tree)


# %%
# decoding is needed to transform the bytes object into an actual string
print(lxml.etree.tostring(tree).decode())


# %%
for rhyme in tree.iterfind('//rhyme'):
    print(f'element: {rhyme.tag} -> {rhyme.text}')


# %%
root = tree.getroot()
print(root.tag)


# %%
print(root.attrib['year'])


# %%
print(len(root))


# %%
children = [child.tag for child in root]


# %%
print('\n'.join(child.text or '' for child in root))


# %%
print(''.join(root[0].itertext()))


# %%
for node in root:
    if node.tag == 'line':
        print(f"line {node.attrib['n']: >2}: {''.join(node.itertext())}")


# %%
with open('data/sonnets/116.txt') as stream:
    text = stream.read()

print(text)


# %%
root = lxml.etree.Element('sonnet')
root.attrib['author'] = 'William Shakespeare'
root.attrib['year'] = '1609'


# %%
tree = lxml.etree.ElementTree(root)
stringified = lxml.etree.tostring(tree)
print(stringified)


# %%
print(type(stringified))


# %%
print(stringified.decode('utf-8'))


# %%
for nb, line in enumerate(open('data/sonnets/116.txt')):
    node = lxml.etree.Element('line')
    node.attrib['n'] = str(nb + 1)
    node.text = line.strip()
    root.append(node)
    # voltas typically, but not always occur between the octave and sextet
    if nb == 8:
        node = lxml.etree.Element('volta')
        root.append(node)


# %%
print(lxml.etree.tostring(tree, pretty_print=True).decode())


# %%
# Loop over all nodes in the tree
for node in root:
    # Leave the volta node alone. A continue statement instructs
    # Python to move on to the next item in the loop.
    if node.tag == 'volta':
        continue
    # We chop off and store verse-final punctuation:
    punctuation = ''
    if node.text[-1] in ',:;.':
        punctuation = node.text[-1]
        node.text = node.text[:-1]
    # Make a list of words using the split method
    words = node.text.split()
    # We split rhyme words and other words:
    other_words, rhyme = words[:-1], words[-1]
    # Replace the node's text with all text except the rhyme word
    node.text = ' '.join(other_words) + ' '
    # We create the rhyme element, with punctuation (if any) in its tail
    elt = lxml.etree.Element('rhyme')
    elt.text = rhyme
    elt.tail = punctuation
    # We add the rhyme to the line:
    node.append(elt)

tree = lxml.etree.ElementTree(root)
print(lxml.etree.tostring(tree, pretty_print=True).decode())


# %%
with open('data/sonnets/116.xml', 'w') as f:
    f.write(
        lxml.etree.tostring(
            root, xml_declaration=True, pretty_print=True, encoding='utf-8').decode())




# %%
root = lxml.etree.Element('sonnet')
# Add an author attribute to the root node
root.attrib['author'] = 'William Shakespeare'
# Add a year attribute to the root node
root.attrib['year'] = '1609'

for nb, line in enumerate(open('data/sonnets/116.txt')):
    line_node = lxml.etree.Element('line')
    # Add a line number attribute to each line node
    line_node.attrib['n'] = str(nb + 1)
    # Make different nodes for words and non-words
    word = ''
    for char in line.strip():
        if char.isalpha():
            word += char
        else:
            word_node = lxml.etree.Element('w')
            word_node.text = word
            line_node.append(word_node)
            word = ''
            char_node = lxml.etree.Element('c')
            char_node.text = char
            line_node.append(char_node)
    # don't forget last word:
    if word:
        word_node = lxml.etree.Element('w')
        word_node.text = word
        line_node.append(word_node)
    rhyme_node = lxml.etree.Element('rhyme')
    # We use xpath to find the final w-element in the line
    # and wrap it in a line element
    rhyme_node.append(line_node.xpath('//w')[-1])
    line_node.replace(line_node.xpath('//w')[-1], rhyme_node)
    root.append(line_node)
    # Add the volta node
    if nb == 8:
        node = lxml.etree.Element('volta')
        root.append(node)

tree = lxml.etree.ElementTree(root)
xml_string = lxml.etree.tostring(tree, pretty_print=True).decode()
# Print a snippet of the tree:
print(xml_string[:xml_string.find("</line>") + 8] + '  ...')




#
# French Theater in XML format
#
# A larger example with XML files
#
#
import os
import lxml.etree
import tarfile
import collections
import matplotlib.pyplot as plt
import numpy as np


# to do if you have only the .tar.gz file and not the folder /data/theatre-classique/
tarf = tarfile.open('data/theatre-classique.tar.gz', 'r')
tarf.extractall('data')

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

#
#
#
print (len(plays), len(genres), len(titles), len(authors), len(years))

#
# Overview Statistics
#
counts = collections.Counter(genres)

fig, ax = plt.subplots()
ax.bar(counts.keys(), counts.values(), width=0.3)
ax.set(xlabel="genre", ylabel="count");
fig.show()


# many names for the same person (authors = list in python)
authors


#
#
#  TEI
#
#

# %%
tree = lxml.etree.parse('data/folger/xml/Oth.xml')
print(tree.getroot().find('.//{http://www.tei-c.org/ns/1.0}title').text)


# %%
print(tree.getroot().find('title'))

NSMAP = {'tei': 'http://www.tei-c.org/ns/1.0’}
print(tree.getroot().find('.//tei:title', namespaces=NSMAP).text)


#
#  HTML format
#
#



# %%
import bs4 as bs

html_doc = """
<html>
  <head>
    <title>Henry IV, Part I</title>
  </head>
  <body>
    <div>
      <p class="speaker">KING</p>
      <p id="line-1.1.1">
        <a id="ftln-0001">FTLN 0001</a>
        So shaken as we are, so wan with care,
      </p>
      <p id="line-1.1.2">
        <a id="ftln-0002">FTLN 0002</a>
        Find we a time for frighted peace to pant
      </p>
      <p id="line-1.1.3">
        <a id="ftln-0003">FTLN 0003</a>
        And breathe short-winded accents of new broils
      </p>
      <p id="line-1.1.4">
        <a id="ftln-0004">FTLN 0004</a>
        To be commenced in strands afar remote.
      </p>
    </div>
  </body>
</html>
"""


html = bs.BeautifulSoup(html_doc, 'html.parser')


# %%
# print the documents <title> (from head)
print(html.title)


# %%
# print the first <p> element and its content
print(html.p)


# %%
# print the text of a particular element, e.g. the <title>
print(html.title.text)


# %%
# print the parent tag (and its content) of the first <p> element
print(html.p.parent)


# %%
# print the parent tag name of the first <p> element
print(html.p.parent.name)


# %%
# find all occurrences of the <a> element
print(html.find_all('a'))


# %%
# find a <p> element with a specific ID
print(html.find('p', id='line-1.1.3'))


# %%
def html2txt(fpath):
    """Convert text from a HTML file into a string.
    Arguments:
        fpath: a string pointing to the filename of the HTML file
    Returns:
        A string representing the text extracted from the supplied
        HTML file.
    """
    with open(fpath) as f:
        html = bs.BeautifulSoup(f, 'html.parser')
    return html.get_text()


# %%
fp = 'data/folger/html/1H4.html'
text = html2txt(fp)
start = text.find('Henry V')
print(text[start:start + 500])


# %%
with open(fp) as f:
    html = bs.BeautifulSoup(f, 'html.parser')

toc = html.find('table', attrs={'class': 'contents'})


# %%
def toc_hrefs(html):
    """Return a list of hrefs from a document's table of contents."""
    toc = html.find('table', attrs={'class': 'contents'})
    hrefs = []
    for tr in toc.find_all('tr'):
        for td in tr.find_all('td'):
            for a in td.find_all('a'):
                hrefs.append(a.get('href'))
    return hrefs


# %%
items = toc_hrefs(html)
print(items[:5])


# %%
def get_href_div(html, href):
    """Retrieve the <div> element corresponding to the given href."""
    href = href.lstrip('#')
    div = html.find('div', attrs={'id': href})
    if div is None:
        div = html.find('a', attrs={'name': href}).findNext('div')
    return div


# %%
def html2txt(fname, concatenate=False):
    """Convert text from a HTML file into a string or sequence of strings.
    Arguments:
        fpath: a string pointing to the filename of the HTML file.
        concatenate: a boolean indicating whether to concatenate the
            extracted texts into a single string. If False, a list of
            strings representing the individual sections is returned.
    Returns:
        A string or list of strings representing the text extracted
        from the supplied HTML file.
    """
    with open(fname) as f:
        html = bs.BeautifulSoup(f, 'html.parser')
    # Use a concise list comprehension to create the list of texts.
    # The same list could be constructed using an ordinary for-loop:
    #    texts = []
    #    for href in toc_hrefs(html):
    #        text = get_href_div(html, href).get_text()
    #        texts.append(text)
    texts = [get_href_div(html, href).get_text() for href in toc_hrefs(html)]
    return '\n'.join(texts) if concatenate else texts


# %%
texts = html2txt(fp)
print(texts[6][:200])


# %%
import urllib.request

page = urllib.request.urlopen('https://en.wikipedia.org/wiki/William_Shakespeare')
html = page.read()


# %%
import bs4

soup = bs4.BeautifulSoup(html, 'html.parser')
print(soup.get_text().strip()[:300])


# %%
import re

for script in soup(['script', 'style']):
    script.extract()
text = soup.get_text()
text = re.sub('\s*\n+\s*', '\n', text)  # remove multiple linebreaks:
print(text[:300])



