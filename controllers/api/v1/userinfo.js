const
	express = require('express'),
	session = require('express-session'),
	validator = require('../../../util/account_info_validation.js'),
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
	var responseJson = {
		errno: 0,
		errstr: '',
	};
	var respond_error = function(errJson, resHeadStatus) {
		if(resHeadStatus) {
			res.statusCode = resHeadStatus;
		}
		res.json(errJson);
		res.end();
	};

	var username = req.body.username;

	if(!username) {
		console.log(username);
		respond_error({errno: 2, errstr: 'Invalid or unknown username'});
		return;
	} else if (!req.session.user_id || req.session.user_id !== username) {
		respond_error({errno: 9, errstr: 'Unauthorized'}, 401);
		return;
	}

	var query_pieces = [];
	var values = {'username': username};
	if(req.body['firstname']) {
		if (!validator.is_valid_firstname(req.body['firstname'])) {
			respond_error({errno: 6, errstr: 'Invalid firstname'});
			return;
		}
		query_pieces.push('firstname = ${firstname}');
		values['firstname'] = req.body['firstname'];
	}
	if(req.body['lastname']) {
		if (!validator.is_valid_lastname(req.body['lastname'])) {
			respond_error({errno: 6, errstr: 'Invalid lastname'});
			return;
		}
		query_pieces.push('lastname = ${lastname}');
		values['lastname'] = req.body['lastname'];
	}
	if(req.body['self_bio']) {
		if (!validator.is_valid_self_bio(req.body['self_bio'])) {
			respond_error({errno: 6, errstr: 'Invalid self_bio'});
			return;
		}
		query_pieces.push('self_bio = ${self_bio}');
		values['self_bio'] = req.body['self_bio'];
	}
	if(req.body['institution']) {
		if (!validator.is_valid_institution(req.body['institution'])) {
			respond_error({errno: 6, errstr: 'Invalid institution'});
			return;
		}
		query_pieces.push('institution = ${institution}');
		values['institution'] = req.body['institution'];
	}

	if(query_pieces.length === 0) {
		respond_error({errno: 7, errstr: 'Missing parameter (one of firstname, lastname, institution, etc)'});
		return;
	}

	var queryStr = 'UPDATE account SET ' + query_pieces.join(', ') + ' WHERE username = ${username};';

	pgdb.none(queryStr, values)
	.then((data) => {
		res.json(responseJson);
		res.end();
	})
	.catch((err) => {
		console.error(err);
		respond_error({errno: 1, errstr: 'Unknown error'});
	});
}


module.exports = {
	getUserInfo: getUserInfo,
	putUserInfo: putUserInfo
};

