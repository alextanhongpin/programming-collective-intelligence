import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a Series by passing a list of values
s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print('s:\n', s)

# Create a DataFrame by passing a numpy of array, with a datetime index and labelled counts
dates = pd.date_range('20130101', periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6, 5), index=dates, columns=list('ABCDE'))
print(df)

# Create a DataFrame by passing a dict of objects that can be converted to series-like
df2 = pd.DataFrame({
    'A': 1.0,
    'B': pd.Timestamp('20130101'),
    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
    'D': np.array([3] * 4, dtype='int32'),
    'E': pd.Categorical(['test', 'train', 'test', 'train']),
    'F': 'foo'
})
print(df2)
# Get all the available types in this DataFrame
print(df2.dtypes)

# See the top rows, defaults to 5
print(df.head(3))

# See the last 3 rows
print(df.tail(3))

# List the index
print(df.index)

# List the columns name
print(df.columns)

# List only the values
print(df.values)

# Shows a quick statistic summary of your data
print(df.describe())

# Transposing your data
print(df.T)

# Sorting by axis
print(df.sort_index(axis=1, ascending=False))

# Sorting by values
print(df.sort_values(by='B'))

# Get a sorted column
print(df.sort_values(by='A')['A'])

# Select the first 3 rows
print(df[0:3])

# Or select by the row values
print(df['20130103':'20130105'])

# For getting a cross section using a label (first row)
print(df.loc['20130101'])

# Selecting on a multi-axis by label
print(df.loc[:,['A', 'B']])

# Showing label slicing, both endpoints included
print(df.loc['20130102':'20130104',['A', 'B']])

# Reduction of dimension of the returned object
print(df.loc['20130101', ['A', 'B']])

# For getting a scalar value
print(df.loc['20130101', 'A'])

# Selection via the position of the passed integer
print(df.iloc[3]) # Prints the item in the fourth row

# By integer slices
print(df.iloc[3:5, 0:2]) # Prints the item for the fourth and fifth row, and only the first two columns

# By list of integer position locations
print(df.iloc[[1,3,4], [0, 2]]) # Prints the second, fourth and fifth rows, and the first and third column

# For slicing rows explicitly
print(df.iloc[1:3, :])

# For slicing columns explicitly
print(df.iloc[:,1:3])

# For getting a value explicitly
print(df.iloc[1,1])

# For getting access to a scala
print(df.iat[1,1])

####################
# Boolean Indexing #
####################

# Using a single column to select data
print(df[df.A > 0])

# Selecting values from a DataFrame where a boolean condition is met
print(df[df > 0])

df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']

print(df2)
# Use isin for filtering
print(df2[df2['E'].isin(['two', 'four'])])

###########
# Setting #
###########

sl = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
print(sl)
# Setting a column automatically aligns the data by the indexes
df['F'] = sl
print(df)

# Setting values by label
df.at[dates[0], 'A'] = 0
print(df)

# Setting values by position
df.iat[0, 1] = 0
print(df)

# Setting by assigning with a numpy array
df.loc[:, 'D'] = np.array([5] * len(df))
print(df)

df2 = df.copy()
df2[df2 > 0] = -df2 
print(df2)
# Filter only specific categories and drop all NA categories
# raw_cat = pd.Categorical(['a', 'b', 'c', 'd', 'e'], categories=['b', 'c', 'e'], ordered=False)
# print(pd.Series(raw_cat).dropna())

################
# Missing Data #
################

# Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1
print(df1)

# To drop any rows that have missing data
print(df1.dropna(how='any'))

# Fill any missing data
print(df1.fillna(value=5))

# To get a boolean mask where values are nan
print(pd.isnull(df1))

##############
# Operations #
##############

print(df.mean())

# Same operation on the other axis
print(df.mean(1))

# Shift moves the value down the rows by 2
s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
print(s)

print(df.sub(s, axis='index'))

#########
# Apply #
#########

# Applying functions to the data
print(df.apply(np.cumsum))

print(df.apply(lambda x: x.max() - x.min()))

################
# Histograming #
################

s = pd.Series(np.random.randint(0, 7, size=10))
print(s)

# Buckets
print(s.value_counts())


##################
# String methods #
##################

s = pd.Series(['A','B','C','Aaba','Baca',np.nan,'CABA','dog','cat'])
print(s.str.lower())

##########
# Concat #
##########

# Concatenating pandas object together with concat()

df = pd.DataFrame(np.random.randn(10, 4))
print(df)

pieces = [df[:3], df[3:7], df[7:]]

print(len(pieces))
print(pd.concat(pieces))

########
# Join #
########

left = pd.DataFrame({
    'key': ['foo', 'foo'],
    'lval': [1, 2]
})

right = pd.DataFrame({
    'key': ['foo', 'foo'],
    'rval': [4, 5]
})

print(pd.merge(left, right, on='key'))

##########
# Append #
##########

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])
print(df)

s = df.iloc[3]
print(s)

print(df.append(s, ignore_index=True))


############
# Grouping #
############

df = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar',
          'foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'two', 'two', 'three',
          'two', 'two', 'one', 'three'],
    'C': np.random.randn(8),
    'D': np.random.randn(8)
})
print(df)

print(df.groupby('A').sum())

#############
# Reshaping #
#############

tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
                     'foo', 'foo', 'qux', 'qux'],
                    ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))

print(tuples)

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])

df2 = df[:4]
print(df2)


# The stack method compresses a level in the DataFrame's column
stacked = df2.stack()
print(stacked)

# Opposite of stacked in unstacked
print(stacked.unstack())
print(stacked.unstack(1))
print(stacked.unstack(0))

################
# Pivot Tables #
################

df = pd.DataFrame({
    'A': ['one', 'one', 'two', 'three'] * 3,
    'B': ['A', 'B', 'C'] * 4,
    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
    'D': np.random.randn(12),
    'E': np.random.randn(12)
})

print(df)

print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))

###############
# Time Series #
###############

rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts.resample('5Min').sum())

# Time zone representation
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)

print(ts)

ts_utc = ts.tz_localize('UTC')
print(ts_utc)

# Convert to another time zone
ts_utc.tz_convert('US/Eastern')
print(ts_utc)

# Converting between time span representation
rng = pd.date_range('1/1/2012', periods=5, freq='M')

ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)

ps = ts.to_period()
print(ps)

print(ps.to_timestamp())

################
# Categoricals #
################

df = pd.DataFrame({
    'id': [1,2,3,4,5,6],
    'raw_grade': ['a', 'b', 'b', 'a', 'a', 'e'] 
})
print(df)

# Convert the raw grades to a categorical data type
df['grade'] = df['raw_grade'].astype('category')
print(df['grade'])

# Rename the categories to a more meaningful names
df['grade'].cat.categories = ['very good', 'good', 'very bad']
print(df['grade'])

# Reorder the categories and simultaneously add the missing categories

df['grade'] = df['grade'].cat.set_categories(['very bad', 'bad', 'medium', 'good', 'very good'])
print(df['grade'])
print(df)

# Sorting is per order of the categories, not lexical order
print(df.sort_values(by='grade'))

# Grouping by a categorical columns shows also empty categories
print(df.groupby('grade').size())

############
# Plotting #
############

ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2012', periods=1000))
ts = ts.cumsum()
print(ts.head())

# Hides immediately
import matplotlib.pyplot as plt
ts.plot()
plt.show(block=True)

# Plot columns with label
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])

df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc='best')
plt.show(block=True)


#######
# CSV #
#######

# df.to_csv('foo.csv')
# df.read_csv('foo.csv')
