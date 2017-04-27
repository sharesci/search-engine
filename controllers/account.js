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

	function onInsertComplete(data){
		responseObj.errno = 0;
		responseObj.errstr = "";
		res.json(responseObj);
		res.end();
	}

	function onInsertFailed(err) {
		if (err.code === '23505') {
			// Violated 'UNIQUE' constraint, so username was already in use
			responseObj.errno = 8;
			responseObj.errstr = "Account already exists";
			res.json(responseObj);
		} else {
			console.error(err);
			responseObj.errno = 1;
			responseObj.errstr = "Unknown error";
			res.json(responseObj);
		}
		res.end();
	}

	valuesPromise = new Promise((resolve, reject) => {values_from_request(req, resolve, reject);});
	valuesPromise.then((values)=>{
		return new Promise((resolve, reject)=>{insertValues(values, resolve, reject);});
	})
	.then(onInsertComplete)
	.catch(onInsertFailed);

	valuesPromise.catch((err) => {
		responseObj.errno = err.errno;
		responseObj.errstr = err.errstr;
		res.json(responseObj);
		res.end();
	});

}

function insertValues(values, resolve, reject) {
	const query = 'INSERT INTO account (username, passhash, firstname, lastname, self_bio, institution) VALUES (${username}, ${passhash}, ${firstname}, ${lastname}, ${self_bio}, ${institution});';
	pgdb.any(query, values)
		.then((data) => {
			resolve(data);
		})
		.catch((err) => {
			reject(err);
		});
}


// Sets up values for insertion into the database
// and validates them. Calls `resolve` with a JSON 
// object containing the values on success, calls 
// `reject` with a JSON object containing error info 
// on failure.
function values_from_request(req, resolve, reject) {
	if(!req.body.password) {
		reject({errno: 6, errstr: 'Missing password'});
		return;
	}
	var passsalt = bcrypt.genSaltSync(10);
	var passhash = bcrypt.hashSync(req.body.password, passsalt);
	var values = {
		'username': req.body.username,
		'passhash': passhash,
		'firstname': req.body.firstname,
		'lastname': req.body.lastname,
		'self_bio': req.body.self_bio,
		'institution': req.body.institution
	};

	for (key in values) {
		if(typeof values[key] === 'undefined') {
			values[key] = null;
		}
	}

	// Validate values
	if (!is_valid_username(values['username'])) {
		reject({errno: 2, errstr: 'Invalid username'});
		return;
	}
	if (!is_valid_password(req.body.password)) {
		reject({errno: 3, errstr: 'Invalid password'});
		return;
	}
	if (!is_valid_firstname(values['firstname'])) {
		reject({errno: 6, errstr: 'Invalid firstname'});
		return;
	}
	if (!is_valid_lastname(values['lastname'])) {
		reject({errno: 6, errstr: 'Invalid lastname'});
		return;
	}
	if (!is_valid_institution(values['institution'])) {
		reject({errno: 6, errstr: 'Invalid institution'});
		return;
	}
	if (!is_valid_self_bio(values['self_bio'])) {
		reject({errno: 6, errstr: 'Invalid self-biography'});
		return;
	}

	resolve(values);
	
}


function is_valid_username(username) {
	if (!username) {
		return false;
	}
	return (typeof username === 'string' && 4 <= username.length);
}

function is_valid_password(password) {
	if (!password) {
		return false;
	}
	return (typeof password === 'string' && 8 <= password.length);
}

function is_valid_firstname(firstname) {
	if (!firstname) {
		return false;
	}
	return (typeof firstname === 'string' && 1 <= firstname.length);
}

function is_valid_lastname(lastname) {
	if (!lastname) {
		return false;
	}
	return (typeof lastname === 'string' && 2 <= lastname.length);
}

function is_valid_self_bio(bio) {
	return true;
}

function is_valid_institution(institution) {
	return true;
}



module.exports = {
	index: index,
	createAction: createAction
};



