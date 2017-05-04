const
	express = require('express'),
	session = require('express-session'),
	validator = require('../../../util/account_info_validation.js'),
	pgdb = require('../../../sharesci-pg-db');

function getUserEmail(req, res) {
	var responseJson = {
		errno: 0,
		errstr: '',
		emails: []
	};

	var username = req.query.username;

	if(!username) {
		responseJson.errno = 2;
		responseJson.errstr = 'Invalid or unknown username';
		res.json(responseJson);
		res.end();
		return;
	}

	pgdb.func('get_user_email_addr', [username])
	.then((data) => {
		responseJson.emails = [];
		for(emailJson of data) {
			responseJson.emails.push(emailJson['email']);
		}
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

function putUserEmail(req, res) {
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

	var values = {
		'_in_username': username,
		'_in_email': req.body['email'],
		'_in_isprimary': (typeof req.body['isprimary'] === 'boolean' && req.body['isprimary'])
	};
	if(!validator.is_valid_email(values['_in_email'])) {
		respond_error({errno: 6, errstr: 'Invalid email'});
		return;
	}
	pgdb.proc('put_user_email_addr', [values['_in_username'], values['_in_email'], values['_in_isprimary']])
	.then((data) => {
		res.json(responseJson);
		res.end();
	})
	.catch((err) => {
		console.error(err);
		respond_error({errno: 1, errstr: 'Unknown error'});
	});
}

function deleteUserEmail(req, res) {
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

	var values = {
		'_in_username': username,
		'_in_email': req.body['email']
	};

	var queryStr = 'DELETE FROM email_addr ' +
		' WHERE account_id = (SELECT _id FROM account WHERE username = ${username}) ' +
		' AND email = ${email};';

	pgdb.proc('del_user_email_addr', [values['_in_username'], values['_in_email']])
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
	getUserEmail: getUserEmail,
	putUserEmail: putUserEmail,
	deleteUserEmail: deleteUserEmail
};

