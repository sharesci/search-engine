const
	express = require('express'),
	bodyParser = require('body-parser'),
	accountController = require('../controllers/account');

var router = express.Router();

router.get('/', accountController.index);

router.post('/create', bodyParser.urlencoded({ extended: true }));
router.post('/create', accountController.createAction);

module.exports = router;

