const
	express = require('express'),
	bodyParser = require('body-parser'),
	loginController = require('../controllers/login');

var router = express.Router();

router.post('/', bodyParser.urlencoded({ extended: true }));
router.post('/', loginController.loginAction);
router.get('/', loginController.loginPage);

module.exports = router;

