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
from http import HTTPStatus
import os
import urllib.parse
import io
import numpy as np
import json
import sys
import time
from VectorizerDocSearchEngine import VectorizerDocSearchEngine

class SearchRequestHandler(http.server.BaseHTTPRequestHandler):
	def do_GET(self):
		"""Serve a GET request."""
		req_parts=urllib.parse.urlparse(self.path)
		if req_parts.path == '/search':
			self._do_search_GET()
		elif req_parts.path == '/related-docs':
			self._do_related_docs_GET()
		elif req_parts.path == '/user-recommendations':
			self._do_user_recommendations_GET()
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
		req_body = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
		doc_ids = []
		if '_id' in req_body:
			doc_ids = list(req_body['_id'])
		self.server.search_engine.notify_new_docs(doc_ids)
		
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
		response = dict()

		if 'q' not in req_body:
			response['status'] = HTTPStatus.UNPROCESSABLE_ENTITY  # code 422
			response['body'] = {'errno': 1, 'errstr': 'Missing "q" parameter in GET querystring'}
		else:
			search_params = self._get_generic_search_params(req_body)
			response = self._do_qs_search(req_body['q'][0], **search_params)

		response_body = json.dumps(response['body']).encode()

		self.send_response(response['status'])
		self.send_header("Content-type", "application/json")
		self.send_header("Content-Length", str(len(response_body)))
		self.end_headers()
		f = self.wfile
		f.write(response_body)
		f.flush()


	def _do_related_docs_GET(self):
		req_body = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
		response = dict()

		if 'docid' not in req_body:
			response['status'] = HTTPStatus.UNPROCESSABLE_ENTITY  # code 422
			response['body'] = {'errno': 1, 'errstr': 'Missing "docid" parameter in GET querystring'}
		else:
			search_params = self._get_generic_search_params(req_body)
			response = self._do_related_doc_search(req_body['docid'][0], **search_params)

		response_body = json.dumps(response['body']).encode()

		self.send_response(response['status'])
		self.send_header("Content-type", "application/json")
		self.send_header("Content-Length", str(len(response_body)))
		self.end_headers()
		f = self.wfile
		f.write(response_body)
		f.flush()


	def _do_user_recommendations_GET(self):
		req_body = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
		response = dict()

		if 'userid' not in req_body:
			response['status'] = HTTPStatus.UNPROCESSABLE_ENTITY  # code 422
			response['body'] = {'errno': 1, 'errstr': 'Missing "userid" parameter in GET querystring'}
		else:
			search_params = self._get_generic_search_params(req_body)
			response = self._do_user_recommendations_search(req_body['userid'][0], **search_params)

		response_body = json.dumps(response['body']).encode()

		self.send_response(response['status'])
		self.send_header("Content-type", "application/json")
		self.send_header("Content-Length", str(len(response_body)))
		self.end_headers()
		f = self.wfile
		f.write(response_body)
		f.flush()


	def _get_generic_search_params(self, req_body):
		params = dict()

		# TODO: Add error checking (e.g., if a param is not a valid int)
		params['max_results'] = int(req_body['maxResults'][0]) if 'maxResults' in req_body else 0
		params['offset'] = int(req_body['offset'][0]) if 'offset' in req_body else 0
		params['getFullDocs'] = True if (('getFullDocs' in req_body) and (req_body['getFullDocs'][0].lower() == 'true')) else False

		return params


	def _do_qs_search(self, query, **kwargs):
		return self._do_generic_search(self.server.search_engine.search_qs, query, **kwargs)


	def _do_related_doc_search(self, docid, **kwargs):
		return self._do_generic_search(self.server.search_engine.search_docid, docid, **kwargs)


	def _do_user_recommendations_search(self, userid, **kwargs):
		return self._do_generic_search(self.server.search_engine.search_userid, userid, **kwargs)


	def _do_generic_search(self, search_function, search_param, **kwargs):
		start_time = time.perf_counter()
		search_results = search_function(search_param, **kwargs)
		results_arr = [{'_id': r[1], 'score': r[0]} for r in search_results]

		search_time = time.perf_counter() - start_time
		print('Finished search in {:0.3f} seconds'.format(search_time))

		response = dict()
		response['status'] = 200
		response['body'] = {
			'errno': 0,
			'errstr': '',
			'numResults': len(results_arr),
			'results': results_arr
		};
		return response


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
	search_engine = VectorizerDocSearchEngine()

	server = SearchServer(config_options, search_engine, ('', cmdargs.port), SearchRequestHandler)
	print('Initialized server. Starting on port {}.'.format(cmdargs.port))
	server.serve_forever()

	conn.close()

