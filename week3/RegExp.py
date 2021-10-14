#
# Example of code to use the re package
#
# Regular Expressions
#

# %%
import re


re.search('film', 'a beautiful film!')

re.match('beau', 'beautiful film!')

re.match('film', 'beautiful film!')

myProg = re.compile('film')
myProg.search('a good film!')
myProg.search('a good movie!')

 
myProg = re.compile('film', re.I)

myProg.search('a good FILM!')

myProg = re.compile('film')
myProg.search('a good FILM!')


re.sub('film', 'movie', 'a good film!')

re.sub('film', 'movie', 'a good film! FILM, and film')

re.sub('film', 'movie', 'a good film! FILM, and film', 0, re.I)

re.sub("n't", " not", "I didn't do this. Ann hasn't it.")

re.sub("'ll", " will", "I'll do it, and we'll sing")


re.findall('film', 'a beautiful film! FILM, and Film')

re.findall('film', 'a good film! FILM, and Film', re.I)


def displayMatch(aMatch):
   if aMatch is None:
      return None
   return '<Match: %r>' % (aMatch.group())

myProg = re.compile('film')

displayMatch(myProg.search('this Film is good film!'))

myProg = re.compile('film')
aRes = myProg.search('this is a good film!')

aRes.group(0)
aRes.start(0)
aRes.end(0)


myProg = re.compile('[Ff]ilm')
displayMatch(myProg.search('this Film is good film!'))

myProg = re.compile('[^0123456789]')
displayMatch(myProg.search('2001Troy!'))


aPat = '[$]? ?[0-9]+\.[0-9]*'

re.findall(aPat, '$2.50, $3 1.345 or .95')
re.findall(aPat, '$2.50, $3.0 1.345 or 0.95')


aTweet = "Weak pathetic Democrat Mayor!! https://t.co/dehIDMwgul"
aMatch = re.search('http[s]?://[A-Za-z0-9/\.]* ?', aTweet, re.I)

if (aMatch):
   aPos = max(aMatch.start()-1, 0)
   aTweet = aTweet[:aPos] + " urllink " + aTweet[aMatch.end():]

aTweet


aPat='<[A-Z][A-Za-z]*>.*</[A-Z][A-Za-z]*>'
re.findall(aPat,'<Top>Title</Top> <Head>Headline</Head>')

aPat='<[A-Z]>.*</[A-Z]>'
re.findall(aPat,'<T>Title</T> Tintin <H>Head</H>')


aPat='<[A-Z]>.*?</[A-Z]>'
re.findall(aPat,'<T>Title</T> Tintin <H>Head</H>')


aPat = '<[A-Za-z]>(.*?)</[A-Za-z]>'
re.findall(aPat,'<T>Title</T>  <H>Head</H>')

re.findall(aPat,'<T>Title</t> tintin <h>Head</h>')


aPat = '[$]?([0-9]+)\.([0-9]*)'
re.findall(aPat, '$2.50, $3 1.345 or .95')

re.findall('[\w]+', 'Jean I le bon.')


aLine = "A computer!!! IBM-360 IBM.360 IBM_360 IBM360."

re.findall('[\w]+', aLine)
re.findall('[\w-]+', aLine)
re.findall('[\w\.-]+', aLine)


aLine = "Peter's book? THE price is $32.90."

re.findall("\w+", aLine)
re.findall("\w+['\w+]*", aLine)

re.findall("[^\w\s]", aLine)
re.findall("[^\w\s]", 'Peter. !!!')
re.findall("[^\w\s]+", 'Peter. !!!')


aLine = "this is Peter's book at $32.90 and Ann's pen."
re.findall("\w+|\$[\d\.]+", aLine)

aLine = "I've Peter's book that didn't cost $32.90, yes."
re.findall("[\w']+|\$[\d\.]+", aLine)


aLine = "this is Peter's book at $32.90 and Ann's pen."
re.findall(r"\w+(?:'\w+)?|[^\w\s]", aLine)
re.findall(r"\w+|\$[\d\.]+|\S+", aLine)

aLine = "A computer!!! IBM-360 IBM.360 IBM_360 IBM360."
re.findall(r"\w+|\$[\d\.]+|\S+", aLine)


aLine = "this costly, greatly and poly book"
aPattern = re.compile(u' (\w{3,})+ly[ ,\.]')
aPattern.findall(aLine)


aString = '<a> didn''t eat/drink this!</a>'

re.search('/', aString)	# no match (meta char)
re.search('\/', aString)	# match
re.search(r'/', aString)	# match (raw string)


displayMatch(re.search(r'/', aString))
   
aString = '<a> didn''t eat\drink this!<\a>'
re.search('\', aString)      # no match
re.search(r'\\', aString)	# match (raw string)
re.search('\\', aString)	# impossible
re.search('\\\\', aString)	# match









