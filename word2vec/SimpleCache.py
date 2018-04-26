import time

class SimpleCache:
	## Constructor.
	#
	# @param max_age
	# <br>  The max age, in seconds, of a cache value before it expires
	#
	def __init__(self, max_age=600):
		self._cache = dict()
		self._max_cache_age = max_age
		self._gc_counter = 0


	def _garbage_collect(self):
		cur_time = time.time()
		keys_to_remove = set()
		for main_key in self._cache:
			to_remove = []
			for entry in self._cache[main_key]:
				if entry['expire_time'] < cur_time:
					to_remove.append(entry)
			# Shortcut when removing all entries
			if len(to_remove) == len(self._cache[main_key]):
				keys_to_remove.add(main_key)
				continue

			for entry in to_remove:
				self._cache[main_key].remove(entry)
		for key in keys_to_remove:
			del self._cache[key]

	
	## Returns a result set if a cached one is found for the given
	# parameters. If there is not a cache hit, returns None.
	#
	def get(self, main_key, extra_keys):
		if main_key not in self._cache:
			return None

		cache_entries = self._cache[main_key]
		for entry in cache_entries:
			# We intentionally only check against the given extra
			# keys here, even if the cache entry itself had more
			# extra keys
			is_match = True
			for key in extra_keys:
				if key not in entry['extra_keys'] or entry['extra_keys'][key] != extra_keys[key]:
					is_match = False
					break
			if not is_match:
				continue

			if entry['expire_time'] <= time.time():
				cache_entries.remove(entry)
				break

			return entry['value']

		return None


	def add(self, main_key, extra_keys, value):
		self._gc_counter += 1
		if 20 < self._gc_counter:
			self._garbage_collect()
			self._gc_counter = 0

		if main_key not in self._cache:
			self._cache[main_key] = []

		self._cache[main_key].append({
			'expire_time': time.time() + self._max_cache_age,
			'extra_keys': extra_keys,
			'value': value
		});
