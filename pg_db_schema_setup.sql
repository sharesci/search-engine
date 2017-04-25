\connect sharesci;

CREATE EXTENSION CITEXT;

CREATE TABLE account (
	_id         SERIAL NOT NULL,
	username    TEXT NOT NULL UNIQUE,
	passhash    TEXT NOT NULL,
	self_bio    TEXT,
	firstname   TEXT NOT NULL,
	lastname    TEXT NOT NULL,
	institution TEXT,

	PRIMARY KEY (_id)
);

CREATE TABLE email_addr (
	account_id  INTEGER NOT NULL,
	email       CITEXT NOT NULL,
	isPrimary   BOOLEAN NOT NULL DEFAULT FALSE,

	PRIMARY KEY (account_id, email),

	FOREIGN KEY (account_id) REFERENCES account(_id)
		ON UPDATE CASCADE
		ON DELETE CASCADE
);

