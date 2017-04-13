var express = require('express');
var bodyParser = require('body-parser');
var https = require('https');
var http = require('http');
var fs = require('fs');

var app = express();

var https_ok = true;
var https_options = {};
try {
	var https_options = {
		key: fs.readFileSync('../certs/site_key.pem'),
		cert: fs.readFileSync('../certs/site_cert.pem')
	};
} catch (err) {
	https_ok = false;
	console.error(err);
}

app.use('/', express.static(__dirname + '/static'));

http.createServer(app).listen(7080);

if (https_ok) {
	https.createServer(https_options, app).listen(7443);
}


