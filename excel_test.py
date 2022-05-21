from pandas import DataFrame

# https://stackoverflow.com/a/42375263

l1 = [1, 2, 3, 4]
l2 = [5, 6, 7, 8]
df = DataFrame({'Stimulus Time': l1, 'Reaction Time': l2})
df.to_excel('test.xlsx', sheet_name='test1', index=False)
df2 = DataFrame({'he': l2, 'wot': l1})
df.add
