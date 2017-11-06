--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Alexander M Rush <srush@seas.harvard.edu>
--          Sumit Chopra <spchopra@fb.com>
--          Jason Weston <jase@fb.com>

-- Script to build the dataset
require('torch')
local utils = require('summary/util')

torch.setdefaulttensortype('torch.LongTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Build torch serialized version of a summarization problem.')
cmd:text()

cmd:option('-window', 5, 'The ngram window to use.')

cmd:option('-inTitleFile', '',       'The input file.')
cmd:option('-inTitleDictionary', '', 'The input dictionary.')
cmd:option('-outTitleDirectory', '', 'The output directory.')
cmd:option('-inArticleFile', '',     'The input file.')
cmd:option('-inArticleDictionary', '', 'The input dictionary.')
cmd:option('-outArticleDirectory', '', 'The output directory.')

opt = cmd:parse(arg)

--[[
      The Counter Data Structure
      {
          nsents :- Represents the total number of sentences in this file. (int)
          line_lengths :- stores the total number of lines of a specific length (dict)
          aligned_lengths :- Stores the lengths of every sentence according to index
          bucket_words :- Basically measures the number of words that belong to sentence of a particular length.
                          The stop word is also included in it (for every sentence).
          max_length :- not being used right now. #TODO, should be removed
      }

--]]
local function count(file, aligned_lengths, pad)
   -- Count up properties of the input file.
   local f = io.open(file, 'r')
   local counter = {
      nsents = 0,
      max_length = 0,
      aligned_lengths = {},
      line_lengths = {},
      bucket_words = {}}
   local nline = 1
   for l in f:lines() do
      local true_l = l
      if pad then
         true_l = "<s> <s> <s> " .. l .. " </s> </s> </s>"
      end
      local line = utils.string_split(true_l, " ")
      counter.line_lengths[#line] = (counter.line_lengths[#line] or 0) + 1
      counter.nsents = counter.nsents + 1
      counter.aligned_lengths[nline] = #line
      if aligned_lengths ~= nil then
         -- Add extra for implicit </s>.
         counter.bucket_words[aligned_lengths[nline]] =
            (counter.bucket_words[aligned_lengths[nline]] or 0)
            + #line + 1
      end
      nline = nline + 1
   end
   return counter
end


local function build_article_matrices(dict, file, nsents, line_lengths)
   -- For each length bucket, construct a #sentence x length matrix
   -- of word forms.
   local f = io.open(file, 'r')

   -- One matrix for each length.
   local mat = {}

   -- Number of sentences seen of this length.
   local of_length = {}

   for length, count in pairs(line_lengths) do
      mat[length] = torch.zeros(count, length):long()
      of_length[length] = 1
   end

   -- For each sentence.
   -- Col 1 is its length bin.
   -- Col 2 is its position in bin.
   local pos = torch.zeros(nsents, 2):long()

   local nsent = 1
   for l in f:lines() do
      local true_l = "<s> <s> <s> " .. l .. " </s> </s> </s>"
      local line = utils.string_split(true_l, " ")
      local length = #line
      local nbin = of_length[length]
      -- For every word in the given line
      for j = 1, #line do
         local index = dict.symbol_to_index[line[j]] or 1
         --assert(index ~= nil)
         mat[length][nbin][j] = index
      end
      pos[nsent][1] = length
      pos[nsent][2] = nbin
      of_length[length] = nbin + 1
      nsent = nsent + 1
   end
   return mat, pos
end

--[[
      1.) The aligned_lengths parameter is from the articles counter.
      2.) The bucket_words param. is from the titles counter.
      3.) #TODO, nsent and nline seem to be redundant. Only one can be used.
      4.) pos matrix is not being used. #TODO Remove it.

--]]

local function build_title_matrices(dict, file, aligned_lengths,
                                    bucket_sizes, window)
   -- For each article length bucket construct a num-words x 1 flat vector
   -- of word forms and a corresponding num-words x window matrix of
   -- context forms.
   local nsent = 1
   local pos = {}

   -- One matrix for each length.
   local mat = {}
   local ngram = {}

   -- Number of sentences seen of this length.
   local sent_of_length = {}
   local words_of_length = {}

   -- Initialize.
   for length, count in pairs(bucket_sizes) do
      mat[length] = torch.zeros(count, 3):long()
      sent_of_length[length] = 1 -- Required for the sentence lengths.
      words_of_length[length] = 1 -- Required for bucket words
      ngram[length] = torch.zeros(count, window):long()
   end

   -- Columns are the preceding window.
   local nline = 1
   local f = io.open(file, 'r')
   for l in f:lines() do
      -- Add implicit </s>.
      local true_l = l .. " </s>"
      local line = utils.string_split(true_l, " ")

      local last = {}
      -- Initialize window as START symbol.
      for w = 1, window do
         table.insert(last, dict.symbol_to_index["<s>"])
      end

      local aligned_length = aligned_lengths[nline]
      for j = 1, #line do
         local nword = words_of_length[aligned_length]
         local index = dict.symbol_to_index[line[j]] or 1

         --[[
              1.) Eg:- There maybe 2 sentences of length 11 and in all they may contain 9 words.
              2.) We consider these words one by one.
              3.) index is that individual word's index in the dictionary.
              4.) Then we specify the sentence this word belongs to.
              5.) Finally, we have the position in the sentence to which this word belongs.
         --]]
         mat[aligned_length][nword][1] = index
         mat[aligned_length][nword][2] = sent_of_length[aligned_length]
         mat[aligned_length][nword][3] = j

         -- Move the window forward. The window is always considered backwards including the current word. 
         for w = 1, window-1 do
            ngram[aligned_length][nword][w] = last[w]
            last[w] = last[w+1]
         end
         ngram[aligned_length][nword][window] = last[window]
         last[window] = index
         words_of_length[aligned_length] = words_of_length[aligned_length] + 1
      end
      sent_of_length[aligned_length] = sent_of_length[aligned_length] + 1
      nsent = nsent + 1

      -- Debug logging.
      if nsent % 100000 == 1 then
         print(nsent)
      end
      nline = nline + 1
   end
   return mat, pos, ngram
end

local function main()
   local counter = count(opt.inArticleFile, nil, true)
   local dict = torch.load(opt.inArticleDictionary)

   -- Construct a rectangular word matrix.
   local word_mat, offset_mat =
      ls (dict, opt.inArticleFile,
                             counter.nsents, counter.line_lengths)
   torch.save(opt.outArticleDirectory .. '/word.mat.torch', word_mat)
   torch.save(opt.outArticleDirectory .. '/offset.mat.torch', offset_mat)

   local title_counter = count(opt.inTitleFile, counter.aligned_lengths, false)
   local title_dict = torch.load(opt.inTitleDictionary)

   -- Construct a 1d word matrix.
   local word_mat, offset_mat, ngram_mat =
      build_title_matrices(title_dict,
                           opt.inTitleFile,
                           counter.aligned_lengths,
                           title_counter.bucket_words,
                           opt.window)
   torch.save(opt.outTitleDirectory .. '/word.mat.torch', word_mat)
   torch.save(opt.outTitleDirectory .. '/offset.mat.torch', offset_mat)
   torch.save(opt.outTitleDirectory .. '/ngram.mat.torch', ngram_mat)
end

main()
