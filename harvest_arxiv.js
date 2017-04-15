var request = require('request'),
	MongoClient = require('mongodb').MongoClient,
	assert = require('assert'),
	cheerio = require('cheerio'),
	fs = require('fs'),
	xml2js = require('xml2js');

// Connection URL
var url = 'mongodb://localhost:27017/sharesci';


function mongoInsertPaperCallback(db, paperdata, callback) {
	var collection = db.collection('papers');
	collection.drop();
	collection.insertMany(paperdata, (err, result) => {
		assert.equal(err, null);
		console.log("Inserted " + result.result.n + " papers");
		callback(result);
	});
}

function mongoInsertPaper(paperdata) {
		MongoClient.connect(url, function(err, db) {
			assert.equal(null, err);
			console.log("Connected successfully to server");
			mongoInsertPaperCallback(db, paperdata, (result)=>{db.close();});
		});
}


request('http://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXiv&from=2017-04-01', function(error, response, html) {
	if (!error && response.statusCode == 200) {
		var xmld = cheerio.load(html, {
			xmlMode: true
		});
		alldata = [];
		var metadatax = xmld('record > metadata > arXiv');
		xmld('categories', metadatax).remove();
		metadatax.each((index, obj) => {
			xml2js.parseString(xmld.xml(obj), {explicitArray: false, trim: true}, (err, data) => {
				data = data['arXiv'];
				delete data['$'];
				data['arXiv_id'] = data['id'];
				delete data['id'];
				data['authors'] = data['authors']['author'];
				data['references'] = [];
				alldata.push(data);
			});
		});
		mongoInsertPaper(alldata);
	}
});

