\connect sharesci;

CREATE EXTENSION IF NOT EXISTS CITEXT;

CREATE TABLE account (
	_id         SERIAL NOT NULL,
	username    CITEXT NOT NULL UNIQUE,
	passhash    TEXT NOT NULL,
	self_bio    TEXT,
	firstname   CITEXT NOT NULL,
	lastname    CITEXT NOT NULL,
	institution CITEXT,

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


CREATE TYPE public_user_info AS (username CITEXT, 
	firstname CITEXT, 
	lastname CITEXT, 
	institution CITEXT, 
	self_bio TEXT);


CREATE OR REPLACE FUNCTION get_user_public_info(in CITEXT) 
RETURNS public_user_info
AS $$
	SELECT username, firstname, lastname, institution, self_bio 
		FROM account 
		WHERE username = $1
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION get_user_email_addr(in _in_username CITEXT)
RETURNS TABLE(email CITEXT,
	isPrimary BOOLEAN)
AS $$
	SELECT email, isPrimary FROM email_addr e INNER JOIN account a ON e.account_id = a._id WHERE a.username = _in_username
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION put_user_email_addr(in _in_username CITEXT, in _in_email CITEXT, in _in_isprimary BOOLEAN DEFAULT FALSE)
RETURNS void
AS $$
	INSERT INTO email_addr (account_id, email, isprimary)
		SELECT account._id, _in_email, _in_isprimary
			FROM account
			WHERE username = _in_username;
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION del_user_email_addr(in _in_username CITEXT, in _in_email CITEXT)
RETURNS void
AS $$
	DELETE FROM email_addr
	WHERE
		email_addr.account_id = (SELECT account._id
			FROM account
			WHERE username = _in_username)
		AND email = _in_email;
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION get_user_passhash(in _in_username CITEXT, out passhash TEXT)
AS $$
	SELECT passhash FROM account WHERE username = _in_username
$$ LANGUAGE SQL;


CREATE OR REPLACE FUNCTION put_user_passhash(in _in_username CITEXT, in _in_passhash TEXT)
RETURNS void
AS $$
	UPDATE account SET passhash = _in_passhash WHERE username = _in_username
$$ LANGUAGE SQL;


