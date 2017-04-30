const
	express = require('express'),
	assert = require('assert'),
	ObjectId = require('mongodb').ObjectId,
	MongoClient = require('mongodb').MongoClient,
	mongo_url = 'mongodb://localhost:27017/sharesci',
	validator = require('../../../util/account_info_validation');


function getArticle(req, res) {
	var responseJson = {
		errno: 0,
		articleJson: null
	};

	var usePdf = (req.query.pdf === "1");

	MongoClient.connect(mongo_url, function(err, db) {
		if(err !== null) {
			console.error("Error opening db");
			res.status(500).json({errno: 1, errstr: 'Error opening DB'}).end();
			return;
		}
		if (!req.query.id) {
			db.close();
			responseJson.errno = 5;
			res.status(422).json(responseJson);
			res.end();
			return;
		}
		var cursor;

		try {
			cursor = db.collection('papers').find({'_id': new ObjectId(req.query.id)});
		} catch(err) {
			console.error(err);
			res.status(500).json({errno: 1, errstr: 'Unknown error'});
			return;
		}
		cursor.toArray((err, articleJson)=>{
			if(err){
				res.writeHead(500);
				responseJson.errno = 1;	
			} else {
				responseJson.errno = 0;
				responseJson.articleJson = articleJson;
			}
			db.close();
			if(!usePdf) {
				res.json(responseJson);
				res.end();
				return;
			}
			if(articleJson.length === 0) {
				res.status(404).json({errno: 1, errstr: 'File not found'}).end();
				return;
			}
			articleJson = articleJson[0];
			if(!articleJson['file'] || !articleJson['file']['name']) {
				res.status(404).json({errno: 1, errstr: 'File not found'}).end();
				return;
			}
			var headers = {
				'x-timestamp': new Date(),
				'x-sent': true
			}
			if(typeof articleJson['file']['mimetype'] === 'string') {
				headers['Content-Type'] = articleJson['file']['mimetype'];
			}
			if(typeof articleJson['file']['originalname'] === 'string') {
				headers['Content-disposition'] = 'inline; filename='+articleJson['file']['originalname'];
			}
			var sendfileOptions = {
				root: __dirname + '/../../../uploads/',
				dotfiles: 'deny',
				headers: headers
			};
			res.sendFile(articleJson['file']['name'], sendfileOptions, (err) => {
				if(err) {
					console.error(err);
					console.error('Problem sending file: ', articleJson);
				}
				res.end();
			});
		});
	});
}

function putArticle(req, res) {
	var responseJson = {
		errno: 0,
		errstr: ''
	};

	var metaJson = req.body.metaJson;
	if(!validator.is_valid_articleMetaJson(metaJson)) {
		console.error('Invalid JSON');
		res.status(422).json({errno: 1, errstr: 'Invalid JSON'}).end();
		return;
	}
	metaJson = JSON.parse(metaJson);

	var curDateJson = (new Date()).toJSON();

	// Sanitize some protected fields
	delete metaJson['file'];
	metaJson['updated'] = curDateJson;

	if(!metaJson['_id']) {
		delete metaJson['_id'];
		if (!metaJson['title']) {
			res.status(422).json({errno: 1, errstr: 'Missing title'}).end();
			return;
		}
		metaJson['references'] = metaJson['references'] || [];
		metaJson['created'] = curDateJson;
		metaJson['fulltext_text'] = metaJson['fulltext_text'] || "";
	}
	else {
		metaJson['_id'] = new ObjectId(metaJson['_id']);
	}

	if(req.files.length > 0) {
		var fileinfo = req.files[0];
		metaJson['file'] = {
			'name': fileinfo['filename'],
			'originalname': fileinfo['originalname'],
			'mimetype': fileinfo['mimetype']
		};
	}

	MongoClient.connect(mongo_url, function(err, db) {
		if(err !== null) {
			console.error("Error opening db");
			res.status(500).json({errno: 1, errstr: 'Error opening DB'}).end();
			return;
		}
		var handlerFunc = (err, data)=>{
				if(err){
					res.writeHead(500);
					responseJson.errno = 1;	
				} else {
					responseJson.errno = 0;
					responseJson.errstr = '';
				}
				if(data['insertedIds']) {
					responseJson.insertedIds = data.insertedIds;
				}
				db.close();
				res.json(responseJson);
				res.end();
			};
		if(metaJson['_id']) {
			var cursor = db.collection('papers').update(queryJson, metaJson, handlerFunc);
		} else {
			var cursor = db.collection('papers').insert(metaJson, handlerFunc);
		}
	});
}


module.exports = {
	getArticle: getArticle,
	putArticle: putArticle
};

