#### 1. Sampling the recipes between 1820-1840 and 1900-1920:

In the given data, I found the total recipes for considered years are 48032. 
However, between 1820-1840 and 1900-1920, I observed that there is a total of 19 years for which recipes are reported. 
We group those sampled recipes year wise as below:


```py
ingredients = df['ingredients'].str.split(';')

# group all rows from the same year
groups = ingredients.groupby('date')

# merge the lists from the same year
ingredients = groups.sum()

# compute counts per year
ingredients = ingredients.apply(pd.Series.value_counts).fillna(0)

# normalise the counts
ingredients = ingredients.divide(recipe_counts, 0)

# filter only required ingredients for 1820-1840 and 1900-1920
edited_rows = ingredients.loc[((ingredients.index >= 1900) & (ingredients.index <= 1920)) | ((ingredients.index >= 1820) & (ingredients.index <= 1840))]
```


2. Chi2 is used for selecting best feature among all feature recipes. 
I obsered here that "loaf sugar" is mostly used in all the periods of begining of 18th century and ending of 19th century with the percent 0.734. Then pearlash, rice water, baking powder, yeast etc., are mostly used recipes in the given period.
Below are the percentile of mostly used recipes between 1900-1920. Water is heavily used then remaining ones. Then butter, salt etc.,
Below are the percentile of mostly used recipes between 1820-1840. In this period salt is mostly used by people then remaining ones recipes .. Then water, butter etc ..,
From the below graph I observed that perlash, indian meal, beer, salt petre, is hevily used in the period 1820-1840 then 1900-1920. 
vanila, ice, soda, baking powder etc., are mostly used in the period 1900-1920. Tomato is mostly used in the early 19th century and ice water late 18th century. Loaf sugar, lemon water, salt, wine, molass, yeast, paste, lemon peel, currant are almost used heavily in both periods.