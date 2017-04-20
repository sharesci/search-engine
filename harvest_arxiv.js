var request = require('request'),
	MongoClient = require('mongodb').MongoClient,
	assert = require('assert'),
	cheerio = require('cheerio'),
	fs = require('fs'),
	xml2js = require('xml2js');

// Connection URL
var mongo_url = 'mongodb://localhost:27017/sharesci';

var first_url = 'http://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXiv&from=2010-01-01';
var resume_url = 'http://export.arxiv.org/oai2?verb=ListRecords&resumptionToken=';


function oaiXmlToJson(oaiXML) {
	alldata = [];
	var metadatax = oaiXML('record > metadata > arXiv');
	oaiXML('categories', metadatax).remove();
	metadatax.each((index, obj) => {
		xml2js.parseString(oaiXML.xml(obj), {explicitArray: false, trim: true}, (err, data) => {
			data = data['arXiv'];
			delete data['$'];
			data['arXiv_id'] = data['id'];
			delete data['id'];
			data['authors'] = data['authors']['author'];
			data['references'] = [];
			alldata.push(data);
		});
	});
	return alldata;
}

function mongoClearPapers(callback) {
	MongoClient.connect(mongo_url, function(err, db) {
		assert.equal(null, err);
		console.log("Connected successfully to server for clearing papers");
		var collection = db.collection('papers');
		collection.drop();
		db.close();
		callback(0);
	});
}

function mongoInsertPapers(paperdata, callback) {
	MongoClient.connect(mongo_url, function(err, db) {
		assert.equal(null, err);
		console.log("Connected successfully to server for inserting papers");
		var collection = db.collection('papers');
		collection.insertMany(paperdata, (err, result) => {
			assert.equal(err, null);
			console.log("Inserted " + result.result.n + " papers");
			db.close();
			callback(result);
		});
	});
}

function harvestOAI(url, last_promise) {
	var reqpromise = new Promise((resolve, reject) => {
		request(url, (err,res,xml)=>{resolve([err, res, xml]);});
	});
	reqpromise.then((vals) => {
		var error = vals[0],
			response = vals[1],
			xml = vals[2];
		
		if (error || response.statusCode != 200) {
			console.error(error, '\nstatusCode = ' + response.statusCode);
			return;
		}
		var xmld = cheerio.load(xml, {
			xmlMode: true
		});
		var resumptionToken = xmld('ListRecords > resumptionToken').text();
		console.log('resumption = ' + resumptionToken);
		console.log('resumptionXML = ' + xmld.html('ListRecords > resumptionToken'));
		var alldata = oaiXmlToJson(xmld);

		var mongoPromise = new Promise((resolve, reject) => {
			last_promise.then(() => {
				mongoInsertPapers(alldata, resolve);
			});
		});
		if (resumptionToken && 0 < resumptionToken.length) {
			setTimeout(()=>{harvestOAI(resume_url + resumptionToken, mongoPromise);}, 30000);
		}
	});
}


(new Promise((resolve,reject)=>{resolve(0);})).then((val)=>{harvestOAI(first_url, Promise.resolve(val));});

