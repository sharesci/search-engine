const
	express = require('express'),
	pgdb = require('../../../sharesci-pg-db');

function getUserInfo(req, res) {
	var responseJson = {
		errno: 0,
		errstr: '',
		userJson: null
	};

	var username = req.query.username;

	if(!username) {
		responseJson.errno = 2;
		responseJson.errstr = 'Invalid or unknown username';
		res.json(responseJson);
		res.end();
		return;
	}

	pgdb.one('SELECT username, firstname, lastname, institution, self_bio FROM account WHERE username = ${username};', {'username': username})
	.then((data) => {
		responseJson.userJson = data;
		res.json(responseJson);
		res.end();
	})
	.catch((err) => {
		console.error(err);
		responseJson.errno = 1;
		responseJson.errstr = 'Unknown error';
		res.json(responseJson);
		res.end();
	});

}


function putUserInfo(req, res) {

}


module.exports = {
	getUserInfo: getUserInfo,
	putUserInfo: putUserInfo
};

