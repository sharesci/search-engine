const
	express = require('express'),
	searchController = require('../../../controllers/api/v1/search');
	articleController = require('../../../controllers/api/v1/article');

var router = express.Router();

router.get('/search', searchController.index);
router.get('/article', articleController.getArticle);

module.exports = router;


