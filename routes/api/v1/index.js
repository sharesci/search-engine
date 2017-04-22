const
	express = require('express'),
	searchController = require('../../../controllers/api/v1/search');

var router = express.Router();

router.get('/search', require('body-parser').urlencoded({extended:true}));
router.get('/search', searchController.index);

module.exports = router;


