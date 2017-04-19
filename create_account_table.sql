\connect sharesci;

CREATE TABLE account (
	_id       SERIAL NOT NULL,
	username  VARCHAR(255) NOT NULL UNIQUE,
	passhash  VARCHAR(1024) NOT NULL,

	PRIMARY KEY (_id)
);

