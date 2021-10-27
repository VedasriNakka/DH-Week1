
import pandas as pd
import numpy as np; np.random.seed(1)  # fix a seed for reproducible random sampling
import matplotlib.pyplot as plt
import scipy.special
import itertools


# only show counts for these words:
words_of_interest = ['upon', 'the', 'state', 'enough', 'while', 'any', 'his', 'were']
df = pd.read_csv('data/federalist-papersNew2.csv', index_col=0)
df[words_of_interest].sample(6)


# %%
# values associated with the column 'AUTHOR' are one of the following:
# {'HAMILTON', 'MADISON', 'JAY', 'HAMILTON OR MADISON',
#  'HAMILTON AND MADISON'}
# essays with the author 'HAMILTON OR MADISON' are the 12 disputed essays.
fed = disputed_essays = df[df['AUTHOR'] == 'Hamilton OR Madison'].index
assert len(disputed_essays) == 12  # there are twelve disputed essays
# numbers widely used to identify the essays
assert set(disputed_essays) == {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63}


# %%
# gather essays with known authorship: the undisputed essays of
# Madison and Hamilton
fed = df_known = df.loc[df['AUTHOR'].isin(('Hamilton', 'Madison'))]
print(df_known['AUTHOR'].value_counts())



df_known.groupby('AUTHOR')['upon'].describe()

df_known.groupby('AUTHOR')['upon'].head()
df_known.groupby('AUTHOR')['upon'].tail()
df_known.groupby('AUTHOR')['upon'].sum()


# %%
df_known.groupby('AUTHOR')['upon'].plot.hist(
    rwidth=0.6, alpha=0.6, range=(0, 22), legend=True);

plt.show()

df_known.groupby('AUTHOR')['upon'].plot.hist(
    rwidth=0.6, alpha=0.6, bins=20, legend=True);

plt.show()




# %%
df_known.groupby('AUTHOR')['upon'].describe()


# %%
# The expression below applies `mean` to a sequence of binary observations
# to get a proportion. For example,
# np.mean([False, False, True]) == np.mean([0, 0, 1]) == 1/3
proportions = df_known.groupby('AUTHOR')['upon'].apply(
    lambda upon_counts: (upon_counts > 0).mean())
print(proportions)


# %%
proportions.plot.bar(rot=0);

plt.show()

# %%
proportions


# %%
df = pd.read_csv('data/federalist-papersNew2.csv', index_col=0)
author = df['AUTHOR']  # save a copy of the author column
df = df.drop('AUTHOR', axis=1)  # remove the author column
df = df.divide(df.sum(axis=0))  # rate per 1 word
df *= 1000  # transform from rate per 1 word to rate per 1,000 words
df = df.round()  # round to nearest integer
df['AUTHOR'] = author  # put author column back
df_known = df[df['AUTHOR'].isin({'Hamilton', 'Madison'})]

df_known.groupby('AUTHOR')['by'].describe()


# %%
df_known.groupby('AUTHOR')['by'].plot.hist(
    alpha=0.6, range=(0, 35), rwidth=0.9, legend=True);
plt.title("Histogram of 'by'")
plt.show()


# %%
print(df_known.loc[df_known['by'] > 30, 'by'])


# %%
# Undocumented code snippet used in chapter (e.g., for figure generation)
assert df_known.loc[df_known['by'] > 30, 'by'].index == 83


# %%
with open('data/federalist-83.txt') as infile:
    text = infile.read()

# a regular expression here would be more robust
by_jury_count = text.count(' by jury')
by_count = text.count(' by ')
word_count = len(text.split())  # crude word count
by_rate = 1000 * (by_count - by_jury_count) / word_count

print('In Federalist No. 83 (by Hamilton), without "by jury", '
      f'"by" occurs {by_rate:.0f} times per 1,000 words on average.')


# %%

def negbinom_pmf(x, r, prob):
    """Negative binomial probability mass function."""
    # In practice this calculation should be performed on the log
    # scale to reduce the risk of numeric underflow.
    return (
        scipy.special.binom(x + r - 1, r - 1)
        * ((prob) ** r)  * (1-prob) ** x
    )

print('Pr(X = 6):', negbinom_pmf(6, r=5, prob=0.5))
print('Pr(X = 14):', negbinom_pmf(14, r=5, prob=0.5))



# %%  Empirial distibution
df_known[df_known['AUTHOR'] == 'Hamilton']['by'].plot.hist(
    range=(0, 35), density=True, rwidth=0.9);

plt.show()


# %%  Empirial distibution
df_known[df_known['AUTHOR'] == 'Hamilton']['by'].plot.hist(
    bins=30, density=True, rwidth=0.9);

plt.show()





# %%
df_known[df_known['AUTHOR'] == 'Hamilton']['by'].describe()

# Uniform

x = pd.Series( np.arange(20) )

x.plot.hist(
    range=(0, 20), bins=20, density=True, rwidth=0.6);

plt.show()


# Binomial
from scipy.stats import binom

n, p = 20, 0.1

fig, ax = plt.subplots(1, 1)
x = np.arange(0, 20)

ax.plot(x, binom.pmf(x, n, p), 'bo', ms=12, label='binom pmf')
ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=2, alpha=0.2)

plt.show()

n, p, x = 4, 1/6, 2
binom.pmf(x, n, p)





# Poisson
from scipy.stats import poisson

fig, ax = plt.subplots(1, 1)
x = np.arange(0, 20)
mu = 2

ax.plot(x, poisson.pmf(x, mu), 'bo', ms=12, label='Poisson pmf')
ax.vlines(x, 0, poisson.pmf(x, mu), colors='b', lw=2)
ax.set_xlabel('Poisson distribution (mu = 2)')
ax.set_ylabel('Probability  (PMF)')

plt.show()

mu, x = 2, 0
poisson.pmf(x, mu)
mu, x = 2, 1
poisson.pmf(x, mu)
mu, x = 2, 2
poisson.pmf(x, mu)





# Negative binomial

from scipy.stats import nbinom

fig, ax = plt.subplots(1, 1)
# n = nb success, p=prob of success 
n, p = 2, 0.1
x = np.arange(0, 100)

ax.plot(x, nbinom.pmf(x, n, p), 'bo', ms=8, label='nbinom pmf')

ax.vlines(x, 0, nbinom.pmf(x, n, p), colors='b', lw=2)
ax.set_title(fr'$n$ = {n}, $prob$ = {p}')

plt.show()


n, p, x = 5, 0.9, 1
nbinom.pmf(x, n, p)


# Hypergeometric

from scipy.stats import hypergeom

# From a large sample of M=100 words in which 20 are 'by', and we select a bag of 30 
[M, n, N] = [100, 20, 30]
rv = hypergeom(M, n, N)
x = np.arange(0, n+1)
pmf_dogs = rv.pmf(x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, pmf_dogs, 'bo')
ax.vlines(x, 0, pmf_dogs, lw=2)
ax.set_xlabel('# of selected words in our bag-of-words of size 30')
ax.set_ylabel('Probability  (PMF)')
plt.show()


# From a sample of N=200 words in which 30 are 'by', and we select a bag of 50
# what is the prob to see k = 6 'by'
N, K, n, k = 200, 30, 50, 6
hypergeom.pmf(k, N, K, n)

N, K, n, k = 200, 30, 50, 2
hypergeom.pmf(k, N, K, n)

# Macron example
N, K, n, k = 18413088, 75493, 1038889, 2897
hypergeom.pmf(k, N, K, n)


# Negative binomial

from scipy.stats import nbinom

x = np.arange(60)
rs, probs = [5, 6.5, 12, 16], [0.6, 0.3333, 0.4]
params = list(itertools.product(rs, probs))
pmfs = [negbinom_pmf(x, r, prob) for r, prob in params]

fig, axes = plt.subplots(4, 3, sharey=True, figsize=(10, 8))
axes = axes.flatten()

for ax, pmf, (r, prob) in zip(axes, pmfs, params):
    ax.bar(x, pmf)
    ax.set_title(fr'$r$ = {r}, $prob$ = {prob}')


plt.tight_layout();

plt.show()

# example
r, p, k = 5, 0.9, 1
nbinom.pmf(k, r, p)



# %%
authors = ('Hamilton', 'Madison')
r_hamilton, prob_hamilton = 5, 0.412
r_madison, prob_madison = 50, 0.8

# observed
fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
df_known.groupby('AUTHOR')['by'].plot.hist(
    ax=axes[0], density=True, range=(0, 35), rwidth=0.9, alpha=0.6,
    title='Hamilton v. Madison (observed)', legend=True)


# model
simulations = 10000
for author, (r, prob) in zip(authors, [(r_hamilton, prob_hamilton),
                                    (r_madison, prob_madison)]):
    pd.Series(negbinom(r, prob, size=simulations)).plot.hist(
        label=author, density=True, rwidth=0.9, alpha=0.6, range=(0, 35), ax=axes[1])

    
axes[1].set_xlim((0, 40))
axes[1].set_title('Hamilton v. Madison (model)')
axes[1].legend()
plt.tight_layout();
plt.show()


# Normal (Gaussian) Distribution

from scipy.stats import norm

fig, ax = plt.subplots(1, 1)
# n = nb success, p=prob of success 
n, p = 2, 0.1
x = np.arange(0, 100)

ax.plot(x, nbinom.pmf(x, n, p), 'bo', ms=8, label='nbinom pmf')

ax.vlines(x, 0, nbinom.pmf(x, n, p), colors='b', lw=2)
ax.set_title(fr'$n$ = {n}, $prob$ = {p}')

plt.show()


#
# with parametric models
#
#
words_of_interest = ['upon', 'the', 'state', 'enough', 'while', 'any', 'his', 'were']
df = pd.read_csv('data/federalist-papers.csv', index_col=0)
df[words_of_interest].sample(6)
df = df.loc[df['AUTHOR'].isin(('HAMILTON', 'MADISON'))]
print(df['AUTHOR'].value_counts())

fed = pd.read_csv('data/federalist-papersNew2.csv', index_col=0)
fed[words_of_interest].sample(6)
fed = fed.loc[fed['AUTHOR'].isin(('Hamilton', 'Madison'))]
print(fed['AUTHOR'].value_counts())


xx = df.groupby('AUTHOR')['his']
x1 = xx.get_group('HAMILTON')
x2 = xx.get_group('MADISON')

for tag, group in xx:
    print('group by: ', tag)
    print(group)



xx.plot.hist(rwidth=0.6, alpha=0.6, bins=20, legend=True)
plt.title("Histogram with 'his'")
plt.xlabel('Occurrences')
plt.ylabel('Frequency')

anHisto = plt.gca()
p = anHisto.patches  # There are 2x20 patches
p[0].get_xy()
p[1].get_xy()
p[2].get_xy()
p[3].get_xy()

p[0].get_height()
plt.show()


xRange2 = np.arange(0, 50, 1)
xx1 = xx.get_group('HAMILTON')
muH = xx1.mean()  # Hamilton
xx3 = list(poisson.pmf(xRange2, muH)*300)

# Assign colors for each sequence and the names
colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
colors = ['#E69F00', '#009E73']

names = ['Hamilton', 'Poisson (Hamilton)']
plt.hist([xx1, xx3], bins=20, color = colors, label=names)
plt.title("Histogram with 'his'")
plt.xlabel('Occurrences')
plt.ylabel('Frequency')
plt.legend(names)
plt.title("Histogram with 'his'")
plt.show()



xx2 = xx.get_group('MADISON')
sumM = sum(xx2)
muM = xx2.mean()  # Madison
xx4 = list(poisson.pmf(xRange2, muM)*sumM)


# Assign colors for each sequence and the names
colors = ['#E69F00', '#009E73']

names = ['Madison', 'Poisson (Madison)']
plt.hist([xx2, xx4], bins=20, color = colors, label=names)
plt.title("Histogram with 'his'")
plt.xlabel('Occurrences')
plt.ylabel('Frequency')
plt.legend(names)
plt.title("Histogram with 'his'")
plt.show()


