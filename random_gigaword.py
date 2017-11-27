articles = []
titles = []
pairs = []
import random

with open("test.article.txt") as art:
    for line in art:
        articles.append(line.strip())

with open("test.title.txt") as titl:
    for line in titl:
        titles.append(line.strip())

for a1, t1 in zip(articles, titles):
    pairs.append((a1, t1))

rand_smpl = [ pairs[i] for i in sorted(random.sample(xrange(len(pairs)), 2000)) ]

random_articles = open("random_gigaword_articles.txt", "w")
random_titles = open("random_gigaword_titles.txt", "w")

for tup in rand_smpl:
    random_articles.write(tup[0]+"\n")
    random_titles.write(tup[1]+"\n")

random_articles.close()
random_titles.close()
