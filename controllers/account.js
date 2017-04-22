const
	pgdb = require('../sharesci-pg-db'),
	bcrypt = require('bcrypt');


function index(req, res) {
	// TODO: change the redirect based on whether the user is logged in
	res.redirect('/login');
}

function createAction(req, res) {
	var responseObj = {
		errno: 0,
		errstr: ""
	};
	var passsalt = bcrypt.genSaltSync(10);
	var passhash = bcrypt.hashSync(req.body.password, passsalt);
	const query = 'INSERT INTO account (username, passhash) VALUES (${username}, ${passhash});';
	var values = {
		'username': req.body.username,
		'passhash': passhash,
	};
	pgdb.any(query, values)
		.then((data) => {
			responseObj.errno = 0;
			responseObj.errstr = "";
			res.json(responseObj);
			res.end();
		})
		.catch((err) => {
			if (err.code === '23505') {
				// Violated 'UNIQUE' constraint, so username was already in use
				responseObj.errno = 2;
				responseObj.errstr = "Incorrect username";
				res.json(responseObj);
			} else {
				console.error(err);
				responseObj.errno = 1;
				responseObj.errstr = "Unknown error";
				res.json(responseObj);
			}
			res.end();
		});
}


module.exports = {
	index: index,
	createAction: createAction
};



