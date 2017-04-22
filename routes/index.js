const
	express = require('express'),
	path = require('path'),
	loginRouter = require('./login'),
	logoutRouter = require('./logout'),
	accountRouter = require('./account'),
	apiRouter = require('./api');

var router = express.Router();

router.get('/', express.static(__dirname + '/../client'));

router.get('/', function(req, res) {
		res.sendFile(path.resolve('client/src/index.html'));
});
router.use('/login', loginRouter);
router.use('/logout', logoutRouter);
router.use('/account', accountRouter);
router.use('/api', apiRouter);

module.exports = router;

