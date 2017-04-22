const
	pgdb = require('../sharesci-pg-db'),
	bcrypt = require('bcrypt');


function loginAction(req, res)  {
	var responseObj = {
		errno: 0,
		errstr: ""
	};
	if(req.session.user_id) {
		responseObj.errno = 4;
		responseObj.errstr = "Already logged in";
		res.json(responseObj);
		res.end();
	}
	const query = 'SELECT passhash FROM account WHERE username = ${username};';
	var values = {
		'username': req.body.username
	};
	pgdb.one(query, values)
		.then((data) => {
			if (bcrypt.compareSync(req.body.password, data['passhash'])) {
				req.session.user_id = req.body.username;
				responseObj.errno = 0;
				responseObj.errstr = "";
				res.json(responseObj);
			} else {
				responseObj.errno = 3;
				responseObj.errstr = "Incorrect password";
				res.json(responseObj);
			}
			res.end();
		})
		.catch((err) => {
			if(err.received === 0) {
				console.log('Invalid username \'' + req.body.username + '\' tried to log in.');
				responseObj.errno = 2;
				responseObj.errstr = "Incorrect username";
			} else {
				console.error(err);
				responseObj.errno = 1;
				responseObj.errstr = "Unknown error";
			}
			res.json(responseObj);
			res.end();
		});
}

function loginPage(req, res) {
	res.redirect('/');
	res.end();
}


module.exports = {
	loginAction: loginAction,
	loginPage: loginPage
};

