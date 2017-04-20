const
	pgdb = require('../sharesci-pg-db'),
	bcrypt = require('bcrypt');


function index(req, res) {
	// TODO: change the redirect based on whether the user is logged in
	res.redirect('/login');
}

function createAction(req, res) {
	var passsalt = bcrypt.genSaltSync(10);
	console.log(passsalt);
	var passhash = bcrypt.hashSync(req.body.password, passsalt);
	const query = 'INSERT INTO account (username, passhash) VALUES (${username}, ${passhash});';
	var values = {
		'username': req.body.username,
		'passhash': passhash,
	};
	pgdb.any(query, values)
		.then((data) => {
			res.redirect('/');
			res.end();
		})
		.catch((err) => {
			console.error(err);
			res.writeHead(500);
			res.end();
		});
}


module.exports = {
	index: index,
	createAction: createAction
};



