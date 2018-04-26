
import numpy as np
import json
import sys
import pymongo
import bson.objectid
from SimpleCache import SimpleCache


## Interface for an inverted index using MongoDB as a backend.
#
class MongoInvertedIndex:
	## Constructor.
	#
	# @param mongo_collection
	# <br>  A PyMongo connection to the MongoDb collection holding the
	#       inverted index.
	#
	def __init__(self, mongo_collection, mongo_special_objects_coll, mongo_doc_vector_coll, doc_vector_type_name, max_blob_size=6000, split_inflation_factor=4):
		self._mongo_coll = mongo_collection
		self._mongo_specialobj = mongo_special_objects_coll
		self._mongo_doc_vector_coll = mongo_doc_vector_coll
		self._doc_vector_type_name = doc_vector_type_name
		self._max_blob_size = max_blob_size
		self._split_inflation_factor = split_inflation_factor

		self._cache = SimpleCache(max_age = 600)


	def _find_index_blob_by_termid(self, term_id):
		main_key = 'termid.' + str(term_id)
		cached_entry = self._cache.get(main_key, {})
		if cached_entry is not None:
			return cached_entry

		result = self._mongo_coll.find_one({'term_id': term_id})
		if result is not None:
			self._cache.add(main_key, {}, result)

		return result


	def get_doc_vector(self, doc_id):
		return self._mongo_doc_vector_coll.find_one({'_id': doc_id, self._doc_vector_type_name: {'$exists': True}}, {self._doc_vector_type_name: True})[self._doc_vector_type_name]


	def get_num_docs(self):
		num_docs = self._mongo_specialobj.find_one({'key': 'inverted_index_num_docs', 'value': {'$exists': True}})
		if num_docs is None:
			return 0
		return num_docs['value']


	## Returns a `dict` of various metadata about a given `term_id`.
	# 
	# The metadata format is not well-defined and should not be relied
	# upon, but might contain info such as how many documents contain the
	# term or what the max TF for the term is.
	#
	def get_term_info(self, term_id):
		first_blob = self._mongo_coll.find_one({'term_id': term_id, 'term_meta': {'$exists': True}}, projection={'term_meta': True})
		if first_blob is None:
			return None
		return first_blob['term_meta']


	## Returns a `TermIterator` for the given `term_id`.
	#
	def get_term_iterator(self, term_id):
		first_blob = self._find_index_blob_by_termid(term_id)
		return TermIterator(self._mongo_coll, first_blob)


	## Splits the given blob into an array of smaller blobs
	# 
	# This method keeps individual index nodes small and helps to
	# circumvent Mongo's max object size. If the size of the given blob is
	# acceptable, this simply returns `[blob]` (i.e., an array containing
	# just the original blob). If not, it splits the blob into several
	# blobs and returns them in an array. Some properties to note about the
	# returned blobs include:
	#
	#     - The first blob in the returned list will have the same `_id` as
	#       the given blob
	#
	#     - The last blob in the returned list will have the same
	#       `next_blob_id` as the given blob
	#
	#     - All blobs except the last one will have `next_blob_id` set to
	#       `None`, so it is the caller's job to add correct IDs after
	#       inserting into Mongo (the IDs are unknown until Mongo assigns
	#       them)
	# 
	#     - All of the blobs will have their metadata updated appropriately
	#       (e.g., `max_val`, `min_val`, `size`, and everything else that
	#       `_update_blob_meta` handles)
	# 
	#
	def _split_blob(self, blob):
		num_new_blobs = len(blob['docs']) // self._max_blob_size

		# Make a few more blobs than necessary, to reduce the need to
		# split again in the future
		if num_new_blobs != 0:
			num_new_blobs *= self._split_inflation_factor

		# Split evenly rather than taking max_blob_sized chunks, to
		# reduce the need to split in the future
		split_indexes = list(np.linspace(0, len(blob['docs'])+1, num_new_blobs+2).astype(int))

		all_docs = blob['docs']
		next_blob_id = blob['next_blob_id']

		# Important: reuse the given blob, since it has the '_id' field
		blob['docs'] = blob['docs'][split_indexes[0]:split_indexes[1]]
		blob['next_blob_id'] = None
		new_blobs = [blob]

		for new_blob_num in range(1, num_new_blobs+1):
			new_docs = all_docs[split_indexes[new_blob_num]:split_indexes[new_blob_num+1]]
			new_blobs.append({
				'docs': new_docs,
				'next_blob_id': None,
				'size': len(new_docs),
				'max_val': float('inf'),
				'min_val': float('-inf')
			})

		new_blobs[-1]['next_blob_id'] = next_blob_id

		# Update the blob metadata fields
		new_blobs = [self._update_blob_meta(b) for b in new_blobs]

		return new_blobs



	def _update_blob_meta(self, blob):
		if len(blob['docs']) != 0:
			blob['max_val'] = blob['docs'][0][1]
			blob['min_val'] = blob['docs'][-1][1]
		else:
			blob['max_val'] = float('inf')
			blob['min_val'] = float('-inf')
		blob['size'] = len(blob['docs'])

		return blob


	## Inserts new blobs, updating the next_blob_id pointers along the way.
	#
	# The `next_blob_id` pointers are updated in-place, so they are
	# available to the caller
	#
	def _insert_new_linked_blobs(self, new_blobs):
		# Insert the blobs, but note that we go in REVERSE. We
		# need to set the `next_blob_id` pointers, so we need
		# to insert later blobs and let Mongo assign their IDs
		# before we can insert the blobs that point to them
		next_blob_id = None
		for i in range(len(new_blobs)):
			# Take the next blob from the back of the list
			new_blob = new_blobs[len(new_blobs)-i-1]

			if next_blob_id is not None:
				new_blob['next_blob_id'] = next_blob_id

			if '_id' in new_blob:
				self._mongo_coll.update({'_id': new_blob['_id']}, new_blob)
			else:
				insertion_result = self._mongo_coll.insert_one(new_blob)
				next_blob_id = insertion_result.inserted_id


	def _insert_doc_val_tuples(self, term_id, doc_val_tuples):
		if len(doc_val_tuples) == 0:
			return

		doc_val_tuples = sorted(doc_val_tuples, key=lambda x: x[1], reverse=True)
		cur_index = 0

		first_blob = self._mongo_coll.find_one({'term_id': term_id})
		if first_blob is None:
			# Important: ONLY the first blob of a term gets the
			# `term_id` field
			first_blob = {
				'term_id': term_id,
				'docs': [],
				'next_blob_id': None,
				'size': 0,
				'max_val': float('inf'),
				'min_val': float('-inf'),
				'term_meta': {'df': 0}
			}
			insert_result = self._mongo_coll.insert_one(first_blob)
			first_blob['_id'] = insert_result.inserted_id

		blob = first_blob
		while cur_index < len(doc_val_tuples):
			first_elem = doc_val_tuples[cur_index]
			first_elem_val = first_elem[1]

			elems_to_add = []
			while cur_index < len(doc_val_tuples) and (blob['min_val'] <= doc_val_tuples[cur_index][1] or blob['next_blob_id'] is None):
				elems_to_add.append(doc_val_tuples[cur_index])
				cur_index += 1

			first_blob['term_meta']['df'] = first_blob['term_meta']['df'] + len(elems_to_add)

			blob['docs'] = sorted(blob['docs'] + elems_to_add, key=lambda item: item[1], reverse=True)

			# Keep track of the next blob pointer before we split
			next_blob_id = blob['next_blob_id']

			# Split the blob if necessary. If the blob is small
			# enough, we'll just get a list that only contains the
			# original blob with its metadata updated
			blobs = self._split_blob(blob)
			blobs = [self._update_blob_meta(b) for b in blobs]

			# Insert/update the new blobs
			self._insert_new_linked_blobs(blobs)

			if blob == first_blob:
				first_blob = blobs[0]

			# If we haven't yet gone through everything we need to
			# insert, move to the next blob
			blob = None
			if cur_index < len(doc_val_tuples) and next_blob_id is not None:
				blob = self._mongo_coll.find_one({'_id': next_blob_id})

		# Update first_blob, since the DF probably changed
		self._mongo_coll.update({'_id': first_blob['_id']}, first_blob)

		


	## Adds the given document vector to the index.
	#
	# @param docs
	# <br>  List of `(doc_id, doc_vector)` tuples, where `doc_id` is an
	#       ObjectId an doc_vector is a list of `(term_id, val)` pairs
	#       representing the TF vector of the document.
	#
	def index_documents(self, docs):

		# Preliminary format check
		for i in range(len(docs)):
			if len(docs[i]) != 2:
				raise ValueError("Each doc should be a 2-tuple of (doc_id, doc_vector)")
			if not isinstance(docs[i][0], bson.objectid.ObjectId):
				raise ValueError("doc_id must be an ObjectId")

		num_skipped = 0

		# dict mapping term to a list of (doc_id, val) tuples to insert
		term_doc_lists = dict()

		# Get the docs into the inverted index format
		for doc in docs:
			doc_id = doc[0]
			doc_vector = doc[1]

			# Update document vectors collection
			try:
				self._mongo_doc_vector_coll.update_one({'_id': doc_id}, {'$set': {self._doc_vector_type_name: doc_vector}}, upsert=True)
			except pymongo.errors.DocumentTooLarge:
				# If we couldn't insert, just skip this doc for now
				num_skipped += 1
				print("ERROR: Failed to insert vector for doc_id {}".format(str(doc_id)))
				continue

			for doc_term in doc_vector:
				term_id = doc_term[0]
				val = doc_term[1]

				if term_id not in term_doc_lists.keys():
					term_doc_lists[term_id] = []

				term_doc_lists[term_id].append((doc_id, val))

		# Add to MongoDB
		for term_id in term_doc_lists.keys():
			self._insert_doc_val_tuples(term_id, term_doc_lists[term_id])

		# Update the doc counter in Mongo
		num_docs = self.get_num_docs()
		self._mongo_specialobj.update_one({'key': 'inverted_index_num_docs'}, {'$set': {'value': num_docs + len(docs) - num_skipped}}, upsert=True)


## Iterator over the document list for a term.
#
# This iterator is returned from an inverted index and generates tuples of
# `(doc_id, val)`, where `doc_id` is a document ID and `val` is the value of
# the document's TF vector for the term being iterated over.
# 
class TermIterator:
	def __init__(self, mongo_coll, first_blob):
		self._mongo_coll = mongo_coll
		self._cur_blob = first_blob
		self._cur_index = -1


	def __iter__(self):
		return self


	def __next__(self):
		if self._cur_blob is None or 'docs' not in self._cur_blob:
			raise StopIteration()

		self._cur_index += 1
		if len(self._cur_blob['docs']) <= self._cur_index:
			# Go to the next blob in the "linked list"
			if self._cur_blob['next_blob_id'] is not None:
				self._cur_blob = self._mongo_coll.find_one({'_id': self._cur_blob['next_blob_id']})
				self._cur_index = 0

			# In case we went to the next blob, re-check the earlier condition
			if self._cur_blob is None or len(self._cur_blob['docs']) <= self._cur_index:
				raise StopIteration()

		cur_doc = self._cur_blob['docs'][self._cur_index]

		return (cur_doc[0], cur_doc[1])

