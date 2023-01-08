import itertools

indices = [*range(7)]
#  for L in range(len(indices) + 1):
#      for subset in itertools.combinations(indices, L):
#          print(subset)

combinations = {
 7: [subset for subset in itertools.combinations(indices, 5)],
 3: [subset for subset in itertools.combinations(indices, 3)]
}

print(len(combinations[3]))
for k in combinations[3]:
    print(k)


comb = combinations[7]

ticks = []
for c in comb:
    word = list('.' * 7)
    for i in c:
        word[i] = 'x'
    ticks.append(word)
    print(word)

# 0    1    2
['x', 'x', 'x', 'x', 'x', '.', '.']
['x', 'x', 'x', 'x', '.', 'x', '.']
['x', 'x', 'x', 'x', '.', '.', 'x']
['x', 'x', 'x', '.', 'x', 'x', '.']
['x', 'x', 'x', '.', 'x', '.', 'x']
['x', 'x', 'x', '.', '.', 'x', 'x']

#                3    4    5
['x', 'x', '.', 'x', 'x', 'x', '.']
['x', '.', '.', 'x', 'x', 'x', 'x']
['x', '.', 'x', 'x', 'x', 'x', '.']
['.', 'x', 'x', 'x', 'x', 'x', '.']
['.', 'x', '.', 'x', 'x', 'x', 'x']

# 0    1                        6
['x', 'x', '.', 'x', 'x', '.', 'x']
['x', 'x', '.', 'x', '.', 'x', 'x']
['x', 'x', '.', '.', 'x', 'x', 'x']

#           2              5    6
['x', '.', 'x', '.', 'x', 'x', 'x']
['.', '.', 'x', 'x', 'x', 'x', 'x']
['.', 'x', 'x', 'x', '.', 'x', 'x']
['x', '.', 'x', 'x', '.', 'x', 'x']
['.', 'x', 'x', '.', 'x', 'x', 'x']

#                3    4         6
['x', '.', 'x', 'x', 'x', '.', 'x']
['.', 'x', 'x', 'x', 'x', '.', 'x']
