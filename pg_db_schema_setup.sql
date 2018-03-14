--\connect sharesci;

CREATE EXTENSION IF NOT EXISTS CITEXT;

CREATE TABLE IF NOT EXISTS account (
	_id         SERIAL NOT NULL,
	username    CITEXT NOT NULL UNIQUE,
	passhash    TEXT NOT NULL,
	self_bio    TEXT,
	firstname   CITEXT NOT NULL,
	lastname    CITEXT NOT NULL,
	institution CITEXT,

	PRIMARY KEY (_id)
);

CREATE TABLE IF NOT EXISTS email_addr (
	account_id  INTEGER NOT NULL,
	email       CITEXT NOT NULL,
	isPrimary   BOOLEAN NOT NULL DEFAULT FALSE,

	PRIMARY KEY (account_id, email),

	FOREIGN KEY (account_id) REFERENCES account(_id)
		ON UPDATE CASCADE
		ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS document (
	_id SERIAL NOT NULL,
	text_id TEXT UNIQUE NOT NULL,
	length DOUBLE PRECISION NOT NULL,
	parent_doc INTEGER,
	"type" SMALLINT NOT NULL DEFAULT 1,

	PRIMARY KEY (_id)
);

CREATE TABLE IF NOT EXISTS term (
	_id SERIAL NOT NULL,
	term TEXT NOT NULL,

	PRIMARY KEY (_id)
);

CREATE UNIQUE INDEX term1 on term (CAST(md5(term) AS uuid));

--CREATE INDEX term1 on term (term, _id);

CREATE TABLE IF NOT EXISTS gram (
	gram_id SERIAL NOT NULL,
	term_id_1 INT NOT NULL,
	term_id_2 INT,

	PRIMARY KEY (gram_id, term_id_1, term_id_2)
--	,FOREIGN KEY (term_id_1) REFERENCES term(_id)
--	,FOREIGN KEY (term_id_2) REFERENCES term(_id)
);

CREATE INDEX gram1 on gram (term_id_1, term_id_2, gram_id);
--CREATE INDEX gram2 on gram (term_id_2, gram_id);

CREATE TABLE IF NOT EXISTS tf (
	gram_id INTEGER NOT NULL,
	doc_id INTEGER NOT NULL,
	lnc DOUBLE PRECISION NOT NULL,

	PRIMARY KEY (gram_id, doc_id)

--	,FOREIGN KEY (doc_id) REFERENCES document(_id)
);

CREATE TABLE IF NOT EXISTS idf (
	gram_id INT UNIQUE NOT NULL,
	df BIGINT NOT NULL
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

CREATE OR REPLACE FUNCTION insert_term(in _bigrams TEXT[][]) RETURNS VOID LANGUAGE PLPGSQL AS $$
DECLARE
        _bigram TEXT[];
BEGIN
        FOREACH _bigram SLICE 1 IN ARRAY _bigrams LOOP
                INSERT INTO term(term) VALUES (_bigram[1]), (_bigram[2]) ON CONFLICT (CAST(md5(term) AS uuid)) DO NOTHING;
        END LOOP;
END
$$;



CREATE OR REPLACE FUNCTION insert_bigram_df(in _bigrams TEXT[][], in dfs NUMERIC[]) RETURNS INT[] LANGUAGE PLPGSQL AS $$
DECLARE
	_bigram_ids INT[][];
	_bigram_id INT[];
	_bigram TEXT[];
	_f_term_id INT;
	_s_term_id INT;
	_gram_id INT;
	_i INT = 1;
	gram_ids INT[];
BEGIN
--	FOREACH _bigram SLICE 1 IN ARRAY _bigrams LOOP
--		INSERT INTO term(term) VALUES(_bigram[1]) ON CONFLICT (term) DO NOTHING RETURNING _id INTO _f_term_id;
--		INSERT INTO term(term) VALUES(_bigram[2]) ON CONFLICT (term) DO NOTHING RETURNING _id INTO _s_term_id;
--
--		IF _f_term_id IS NULL THEN
--			SELECT _id FROM term INTO _f_term_id WHERE term.term = _bigram[1];
--		END IF;
--		
--		IF _s_term_id IS NULL THEN
--			SELECT _id FROM term INTO _s_term_id WHERE term.term = _bigram[2];
--		END IF;
--
--		IF _bigram_ids IS NULL THEN
--			_bigram_ids = ARRAY[ARRAY[_f_term_id, _s_term_id]];
--		ELSE
--			_bigram_ids = _bigram_ids || ARRAY[_f_term_id, _s_term_id];
--		END IF;
--	END LOOP;
        
        FOREACH _bigram SLICE 1 IN ARRAY _bigrams LOOP	
--	FOREACH _bigram_id SLICE 1 IN ARRAY _bigram_ids LOOP
                SELECT _id INTO _f_term_id FROM term WHERE term = _bigram[1] AND md5(term)::uuid = md5(_bigram[1])::uuid;
		SELECT _id INTO _s_term_id FROM term WHERE term = _bigram[2] AND md5(term)::uuid = md5(_bigram[2])::uuid;
                RAISE NOTICE 'Returned _f_term_id = (%)', _f_term_id;
                RAISE NOTICE 'Returned _s_term_id = (%)', _s_term_id;

		IF _s_term_id IS NOT NULL THEN
                    SELECT g.gram_id INTO _gram_id
                    FROM gram g
		    WHERE g.term_id_1 = _f_term_id AND g.term_id_2 = _s_term_id;
		ELSE
		    SELECT g.gram_id INTO _gram_id
                    FROM gram g
                    WHERE g.term_id_1 = _f_term_id;
		END IF;

		IF _gram_id IS NULL THEN
			INSERT INTO gram(term_id_1, term_id_2) VALUES (_f_term_id, _s_term_id) RETURNING gram_id INTO _gram_id;
			INSERT INTO idf(gram_id, df) VALUES(_gram_id, dfs[_i]);
		ELSE
			UPDATE idf SET (df) = (idf.df + dfs[_i]) WHERE idf.gram_id = _gram_id;
		END IF;
		gram_ids = array_append(gram_ids, _gram_id);
		_i = _i + 1;
	END LOOP;
	RETURN gram_ids;
END
$$;

