const
	express = require('express'),
	bodyParser = require('body-parser'),
	express_session = require('express-session'),
	https = require('https'),
	http = require('http'),
	fs = require('fs'),
	pg = require('pg-promise')(),
	bcrypt = require('bcrypt'),
	router = require('./config/routes.js');

const app = express();
const pgclient = pg({
	'host': 'localhost',
	'port': 5432,
	'user': 'sharesci',
	'password': 'sharesci',
	'database': 'sharesci',
});

var https_ok = true;
var https_options = {};
try {
	var https_options = {
		key: fs.readFileSync('../certs/site_key.pem'),
		cert: fs.readFileSync('../certs/site_cert.pem')
	};
} catch (err) {
	https_ok = false;
	if (err.errno === -13 && err.syscall === 'open') {
		console.error('Access permissions denied to SSL certificate files.' +
			' HTTPS will not be available. Try running as root.');
	} else {
		console.error(err);
	}
}

app.use('/', express.static(__dirname + '/static'));
app.use(express_session({
	secret: require('crypto').randomBytes(64).toString('base64'),
	resave: false,
	saveUninitialized: false,
	httpOnly: true,
	secure: true,
	ephemeral: true,
	cookie: { maxAge: 60*1000 }
}));


app.use('/login', bodyParser.urlencoded({ extended: true }));
app.post('/login', (req, res) => {
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
	pgclient.one(query, values)
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
});

app.use('/logout', bodyParser.urlencoded({ extended: true }));
app.use('/logout', (req, res) => {
	delete req.session['user_id'];
	req.session.destroy();
	if (req.body.successRedirect) {
		res.redirect(req.body.successRedirect);
	} else {
		res.redirect('/');
	}
});

app.use('/new', bodyParser.urlencoded({ extended: true }));
app.use('/new', (req, res) => {
	var passsalt = bcrypt.genSaltSync(10);
	console.log(passsalt);
	var passhash = bcrypt.hashSync(req.body.password, passsalt);
	const query = 'INSERT INTO account (username, passhash) VALUES (${username}, ${passhash});';
	var values = {
		'username': req.body.username,
		'passhash': passhash,
	};
	pgclient.any(query, values)
		.then((data) => {
			res.redirect('/');
		})
		.catch((err) => {
			console.error(err);
		});
});

function check_login(user, pass, callback) {
}


app.use('/', router);
app.set('views', __dirname + '/src/views');
app.set('view engine', 'ejs');

http.createServer(app).listen(9080);

if (https_ok) {
	https.createServer(https_options, app).listen(9443);
}
