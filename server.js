const
	express = require('express'),
	express_session = require('express-session'),
	https = require('https'),
	http = require('http'),
	fs = require('fs'),
	rootRouter = require('./routes/index');

const app = express();

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

app.use(express_session({
	secret: require('crypto').randomBytes(64).toString('base64'),
	resave: false,
	saveUninitialized: false,
	httpOnly: true,
	secure: true,
	ephemeral: true,
	cookie: { maxAge: 60*1000 }
}));
app.use('/', rootRouter);
app.use('/', express.static(__dirname + '/client'));

http.createServer(app).listen(7080);

if (https_ok) {
	https.createServer(https_options, app).listen(7443);
}
