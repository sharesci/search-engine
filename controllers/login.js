const
	pgdb = require('../sharesci-pg-db'),
	bcrypt = require('bcrypt');


function loginAction(req, res)  {
	if(req.session.user_id) {
		if (req.body.successRedirect) {
			res.redirect(req.body.successRedirect);
		} else {
			res.redirect('/');
		}
	}
	const query = 'SELECT passhash FROM account WHERE username = ${username};';
	var values = {
		'username': req.body.username
	};
	pgdb.one(query, values)
		.then((data) => {
			if (bcrypt.compareSync(req.body.password, data['passhash'])) {
				req.session.user_id = req.body.username;
				if (req.body.successRedirect) {
					res.redirect(req.body.successRedirect);
				} else {
					res.redirect('/');
				}
			} else if (req.body.failureRedirect) {
				res.redirect(req.body.failureRedirect);
			} else {
				res.redirect('/');
			}
			res.end();
		})
		.catch((err) => {
			if(err.received === 0) {
				console.log('Invalid username \'' + req.body.username + '\' tried to log in.');
			} else {
				console.error(err);
			}
			if (req.body.failureRedirect) {
				res.redirect(req.body.failureRedirect);
			} else {
				res.redirect('/');
			}
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

