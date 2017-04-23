const
	express = require('express'),
	apiV1Router = require('./v1');

var router = express.Router();

router.use('/v1', apiV1Router);

module.exports = router;

