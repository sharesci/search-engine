const
	express = require('express'),
	bodyParser = require('body-parser'),
	searchController = require('../../../controllers/api/v1/search'),
	articleController = require('../../../controllers/api/v1/article'),
	userinfoController = require('../../../controllers/api/v1/userinfo');

var router = express.Router();

router.get('/search', searchController.index);
router.get('/article', articleController.getArticle);
router.get('/userinfo', userinfoController.getUserInfo);

router.post('/userinfo', bodyParser.urlencoded({ extended: true }));
router.post('/userinfo', userinfoController.putUserInfo);

module.exports = router;


