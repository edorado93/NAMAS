#
#  Copyright (c) 2015, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
#  Author: Alexander M Rush <srush@seas.harvard.edu>
#          Sumit Chopra <spchopra@fb.com>
#          Jason Weston <jase@fb.com>

"""
Pull out elements of the title-article file.
"""
import sys
#@lint-avoid-python-3-compatibility-imports

# The dictionary input
words_dict = set([l.split()[0]
                  for l in open(sys.argv[2])])

# Input redirection for *.data.txt eg: train.data.filter.txt
for l in sys.stdin:
    splits = l.strip().split("\t")
    if len(splits) != 4:
        continue
    title_parse, article_parse, title, article = l.strip().split("\t")
    if sys.argv[1] == "src":
        print(article)
    elif sys.argv[1] == "trg":
        print(title)
    elif sys.argv[1] == "src_lc":
        words = [w if w in words_dict else "<unk>"
                 for w in article.lower().split()]
        print(" ".join(words))
    elif sys.argv[1] == "trg_lc": 
        words = [w if w in words_dict else "<unk>"
                 for w in title.lower().split()
                 if w not in ['"', "'", "''", "!", "=", "-",
                              "--", ",", "?", ".",
                              "``", "`", "-rrb-", "-llb-", "\\/"]]
        print(" ".join(words))
    elif sys.argv[1] == "srctree":
        print(article_parse)
    elif sys.argv[1] == "interleave":
        # Format needed for T3
        print(article_parse)
        print(title_parse)
