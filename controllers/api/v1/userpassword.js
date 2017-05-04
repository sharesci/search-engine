const
	express = require('express'),
	session = require('express-session'),
	validator = require('../../../util/account_info_validation.js'),
	pgdb = require('../../../sharesci-pg-db'),
	bcrypt = require('bcrypt');


function putUserPassword(req, res) {
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

	if (!validator.is_valid_password(req.body.newPassword)) {
		respond_error({errno: 3, errstr: 'Invalid new password'});
		return;
	}
	if (!req.body.curPassword || typeof req.body.curPassword !== 'string') {
		respond_error({errno: 3, errstr: 'Invalid old password'});
		return;
	}

	var authPromise = pgdb.func('get_user_passhash', [req.body.username])
		.catch((err) => {
			if(err.received === 0) {
				console.log('Invalid username \'' + req.body.username + '\' tried to log in.');
				responseJson.errno = 2;
				responseJson.errstr = "Incorrect username";
			} else {
				console.error(err);
				responseJson.errno = 1;
				responseJson.errstr = "Unknown error";
			}
			res.json(responseJson);
			res.end();
			return Promise.reject(err);
		})
		.then((data) => {
			if (!bcrypt.compareSync(req.body.curPassword, data['passhash'])) {
				responseJson.errno = 3;
				responseJson.errstr = "Incorrect password";
				res.json(responseJson);
				res.end();
				return Promise.reject(responseJson);
			}
			return Promise.resolve({errno: 0, errstr: ''});
		});

	var passsalt = bcrypt.genSaltSync(10);
	var passhash = bcrypt.hashSync(req.body.newPassword, passsalt);
	authPromise.then((data) => {
		return pgdb.proc('put_user_passhash', [req.body.username, passhash]);
	})
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
	putUserPassword: putUserPassword
};


