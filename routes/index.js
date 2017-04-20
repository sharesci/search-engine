const
	express = require('express'),
	loginRouter = require('./login');
	logoutRouter = require('./logout');
	accountRouter = require('./account');

var router = express.Router();

router.get('/', express.static(__dirname + '/../static'));
router.use('/login', loginRouter);
router.use('/logout', logoutRouter);
router.use('/account', accountRouter);

module.exports = router;

