#!/usr/bin/python3

# This file is part of Photos-vectorizer.
#
# Copyright (C) 2017  Mike D'Arcy
#
# Photos-vectorizer is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Photos-vectorizer is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License
# for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import version_check
version_check.assert_min_version('3.5')

import http.server
import os
import urllib.parse
import io
import numpy as np
import json
import sys

class SearchRequestHandler(http.server.BaseHTTPRequestHandler):
	def do_GET(self):
		"""Serve a GET request."""
		req_parts=urllib.parse.urlparse(self.path)
		if req_parts.path == '/search':
			self._do_search_GET()
		else:
			self._do_default_POST()

	def do_POST(self):
		req_parts=urllib.parse.urlparse(self.path)
		if req_parts.path == '/notifynewdoc':
			self._do_notifynewdoc_POST()
		else:
			self._do_default_POST()

	def _do_default_POST(self):
		response_body = json.dumps({'errstr': "There's nothing here. Double-check that you are accessing the right endpoint"})
		self.send_response(404)
		self.send_header("Content-type", "application/json")
		self.send_header("Content-Length", str(len(response_body)))
		self.end_headers()
		f = self.wfile
		f.write(response_body.encode())
		f.flush()

	def _do_notifynewdoc_POST(self):
		# TODO: Add code to update search engine with new doc
		
		# No failure cases implemented yet, so just assume it worked
		response_body = json.dumps({'errstr': ''})
		self.send_response(200)
		self.send_header("Content-type", "application/json")
		self.send_header("Content-Length", str(len(response_body)))
		self.end_headers()
		f = self.wfile
		f.write(response_body.encode())
		f.flush()


	def _do_search_GET(self):
		req_body = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

		max_results = 10 if 'maxResults' not in req_body else req_body['maxResults']

		response = self._do_qs_search(req_body['q'], max_results=max_results)
		response_body = json.dumps(response['body']).encode()

		self.send_response(response['status'])
		self.send_header("Content-type", "application/json")
		self.send_header("Content-Length", str(len(response_body)))
		self.end_headers()
		f = self.wfile
		f.write(response_body)
		f.flush()


	def _do_qs_search(self, query, max_results=0):
		# TODO: Fill in the code to actually search
		return {
			'status': 200,
			'body': {
				"errno": 0,
				"errstr": "string",
				"numResults": 0,
				"results": [
					{
						"_id": "string",
						"documentJson": {},
						"score": 0
					}
				]
			}
		};


class SearchServer(http.server.HTTPServer):
	def __init__(self, config_options, search_engine, *args, **kwargs):
		self.config_options = config_options
		self.search_engine = search_engine
		super(SearchServer, self).__init__(*args, **kwargs)


if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('--port', dest='port', type=int, action='store', default=8000, help='Port to run the server on')
	cmdargs = parser.parse_args(sys.argv[1:])

	# TODO: Add initializations for config_options and search_engine
	config_options = None
	search_engine = None

	server = SearchServer(config_options, search_engine, ('', cmdargs.port), SearchRequestHandler)
	server.serve_forever()

	conn.close()

