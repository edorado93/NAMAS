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

-- Load data for summary experiments.
local util = require('summary/util')

local data = {}

function data.add_opts(cmd)
   cmd:option('-articleDir', '',
              'Directory containing article training matrices.')
   cmd:option('-titleDir', '',
              'Directory containing title training matrices.')
   cmd:option('-validArticleDir', '',
              'Directory containing article matricess for validation.')
   cmd:option('-validTitleDir', '',
              'Directory containing title matrices for validation.')
end

function data.load(article_dir, title_dir)
   return data.init()
end

-- http://lua-users.org/wiki/MetamethodsTutorial
-- The __index method can also be used to track accesses to the table. One of its many use cases
function data.init(title_data, article_data)
   local new_data = {}
   setmetatable(new_data, { __index = data })
   new_data.title_data = title_data
   new_data.article_data = article_data
   new_data:reset()
   return new_data
end

function data:reset()
   self.bucket_order = {} -- This is an array and not a dictionary
   for length, _ in pairs(self.title_data.target) do -- target is the first column of various lengths for the words matrix. So, it's an array of 1D tensors
      table.insert(self.bucket_order, length)
   end
   util.shuffleTable(self.bucket_order)
   self.bucket_index = 0
   self:load_next_bucket()
end

function data:load_next_bucket()
   self.done_bucket = false
   self.bucket_index = self.bucket_index + 1
   self.bucket = self.bucket_order[self.bucket_index] -- Get the article sentence length at this index
   self.bucket_size = self.title_data.target[self.bucket]:size(1) -- Number of sentences of the given length
   self.pos = 1
   self.aux_ptrs = self.title_data.sentences[self.bucket]:float():long()
   -- Column Tensor -> Row Tensor -> Change dimension from 1 * C to 1000 * C and take contiguous i.e no new copy
   -- Creates a matrix of 1000 * length with values 1-length in each row and adds 200*length to each of them
   self.positions = torch.range(1, self.bucket):view(1, self.bucket)
      :expand(1000, self.bucket):contiguous():cuda() + (200 * self.bucket)
end

function data:is_done()
   return self.bucket_index >= #self.bucket_order - 1 and
      self.done_bucket
end

function data:next_batch(max_size)
   local diff = self.bucket_size - self.pos
   if self.done_bucket or diff == 0 or diff == 1 then
      self:load_next_bucket()
   end
   local offset
   if self.pos + max_size > self.bucket_size then
      offset = self.bucket_size - self.pos
      self.done_bucket = true
   else
      offset = max_size
   end
   local positions = self.positions:narrow(1, 1, offset)

   local aux_rows = self.article_data.words[self.bucket]:
      index(1, self.aux_ptrs:narrow(1, self.pos, offset))
   local context = self.title_data.ngram[self.bucket]
      :narrow(1, self.pos, offset)
   local target = self.title_data.target[self.bucket]
      :narrow(1, self.pos, offset)
   self.pos = self.pos + offset
   return {aux_rows, positions, context}, target
end

function data.make_input(article, context, K)
   local bucket = article:size(1)
   local aux_sentence = article:view(bucket, 1)
      :expand(article:size(1), K):t():contiguous():cuda()
   local positions = torch.range(1, bucket):view(bucket, 1)
      :expand(bucket, K):t():contiguous():cuda() + (200 * bucket)
   return {aux_sentence, positions, context}
end

function data.load_title_dict(dname)
   return torch.load(dname .. 'dict')
end

-- ngram[0] = torch.mul(torch.rand(3, 5),100):long() Integer random Tensor
function data.load_title(dname, shuffle, use_dict)
   local ngram = torch.load(dname .. 'ngram.mat.torch')
   local words = torch.load(dname .. 'word.mat.torch')
   local dict = use_dict or torch.load(dname .. 'dict')
   local target_full = {}
   local sentences_full = {}
   local pos_full = {}
   for length, mat in pairs(ngram) do
      if shuffle ~= nil then
         -- :size(1) means the row in row * column
         local perm = torch.randperm(ngram[length]:size(1)):long()
         -- perm gives a LongTensor eg :- [3, 1, 2]. index operation moves the 3rd index to 1st, 1st to 2nd and 2nd to 3rd.
         -- i.e. move according to the Tensor.
         ngram[length] = ngram[length]:index(1, perm):float():cuda() --  The cuda() function returns a CudaTensor copy.
         words[length] = words[length]:index(1, perm)
      else
         ngram[length] = ngram[length]:float():cuda()
      end
      assert(ngram[length]:size(1) == words[length]:size(1))
      --  [{{}, 1}] means all the rows for the first column. Basically it means a[i][1] for all i
      target_full[length] = words[length][{{}, 1}]:contiguous():float():cuda()
      sentences_full[length] = words[length][{{}, 2}]:contiguous():float():cuda()
      pos_full[length] = words[length][{{}, 3}]

   end
   local title_data = {ngram = ngram,
                       target = target_full,
                       sentences = sentences_full,
                       pos = pos_full,
                       dict = dict}
   return title_data
end

function data.load_article(dname, use_dict)
   local input_words = torch.load(dname .. 'word.mat.torch')
   -- local offsets = torch.load(dname .. 'offset.mat.torch')

   local dict = use_dict or torch.load(dname .. 'dict')
   for length, mat in pairs(input_words) do
      input_words[length] = mat
      input_words[length] = input_words[length]:float():cuda()
   end
   local article_data = {words = input_words, dict = dict}
   return article_data
end

return data
