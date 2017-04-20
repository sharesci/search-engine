const pgp = require('pg-promise')();

const pgpdb = pgp({
	'host': 'localhost',
	'port': 5432,
	'user': 'sharesci',
	'password': 'sharesci',
	'database': 'sharesci',
});

module.exports = pgpdb;

