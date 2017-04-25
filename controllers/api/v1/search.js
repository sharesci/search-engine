const
	express = require('express'),
	assert = require('assert'),
	MongoClient = require('mongodb').MongoClient,
	mongo_url = 'mongodb://localhost:27017/sharesci';

function index(req, res) {
	var responseJSON = {
		errno: 0,
		results: []
	};

	var searchParams = JSON.parse(JSON.stringify(req.query));
	if(!searchParams.offset) {
		searchParams.offset = 0;
	}
	if(!searchParams.any) {
		searchParams['any'] = 'estimation';
	}
	if(searchParams.maxResults) {
		searchParams.maxResults = parseInt(searchParams.maxResults);
	}
	var searchPromise = new Promise((resolve, reject)=>{doSearch(searchParams, resolve, reject);});
	searchPromise.then((results) => {
		responseJSON.results = results;
		res.json(responseJSON);
		res.end();
	})
	.catch((err)=>{
		responseJSON.errno = 1;
		res.json(responseJSON);
		res.end();
	});
}


function doSearch(params, resolve, reject) {	
	MongoClient.connect(mongo_url, function(err, db) {
		if(err !== null) {
			console.error("Error opening db");
			reject(err);
			return;
		}
		
		var cursor = db.collection('papers').find({'$and':[{'$text': {'$search': params.any}}]}, {'_id': 1, 'title': 1, score: {'$meta': 'textScore'}}).sort({'score': {'$meta': 'textScore'}}).skip(parseInt(params.offset));
		if(params.maxResults) {
			cursor.limit(params.maxResults);
		}
		cursor.toArray((err, arr)=>{
			console.log(err);
			if(err){
				reject(err);
			} else {
				resolve(arr);
			}
			db.close();
		});
	});
}


module.exports = {
	index: index
};

