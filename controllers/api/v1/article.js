const
	express = require('express'),
	assert = require('assert'),
	ObjectId = require('mongodb').ObjectId,
	MongoClient = require('mongodb').MongoClient,
	mongo_url = 'mongodb://localhost:27017/sharesci';


function getArticle(req, res) {
	var responseJson = {
		errno: 0,
		articleJson: null
	};

	MongoClient.connect(mongo_url, function(err, db) {
		if(err !== null) {
			console.error("Error opening db");
			reject(err);
			return;
		}
		if (!req.query.id) {
			db.close();
			res.writeHead(422);
			responseJson.errno = 5;
			res.json(responseJson);
			res.end();
			return;
		}
		var cursor = db.collection('papers').find({'_id': new ObjectId(req.query.id)});
		cursor.toArray((err, articleJson)=>{
			if(err){
				res.writeHead(500);
				responseJson.errno = 1;	
			} else {
				responseJson.errno = 0;
				responseJson.articleJson = articleJson;
			}
			db.close();
			res.json(responseJson);
			res.end();
		});
	});
}


module.exports = {
	getArticle: getArticle
};

