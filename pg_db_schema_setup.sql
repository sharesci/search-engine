\connect sharesci2;

CREATE EXTENSION IF NOT EXISTS CITEXT;


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
	term TEXT UNIQUE NOT NULL,

	PRIMARY KEY (_id)
);

CREATE TABLE IF NOT EXISTS gram (
	gram_id SERIAL NOT NULL,
	term_id_1 INT NOT NULL,
	term_id_2 INT,

	PRIMARY KEY (gram_id, term_id_1, term_id_2),
	FOREIGN KEY (term_id_1) REFERENCES term(_id),
	FOREIGN KEY (term_id_2) REFERENCES term(_id)
);

CREATE TABLE IF NOT EXISTS tf (
	gram_id INTEGER NOT NULL,
	doc_id INTEGER NOT NULL,
	lnc DOUBLE PRECISION NOT NULL,

	PRIMARY KEY (gram_id, doc_id),

	FOREIGN KEY (doc_id) REFERENCES document(_id)
);

CREATE TABLE IF NOT EXISTS idf (
	gram_id INT UNIQUE NOT NULL,
	df BIGINT NOT NULL
);


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
	FOREACH _bigram SLICE 1 IN ARRAY _bigrams LOOP
		INSERT INTO term(term) VALUES(_bigram[1]) ON CONFLICT (term) DO NOTHING RETURNING _id INTO _f_term_id;
		INSERT INTO term(term) VALUES(_bigram[2]) ON CONFLICT (term) DO NOTHING RETURNING _id INTO _s_term_id;

		IF _f_term_id IS NULL THEN
			SELECT _id FROM term INTO _f_term_id WHERE term.term = _bigram[1];
		END IF;
		
		IF _s_term_id IS NULL THEN
			SELECT _id FROM term INTO _s_term_id WHERE term.term = _bigram[2];
		END IF;

		IF _bigram_ids IS NULL THEN
			_bigram_ids = ARRAY[ARRAY[_f_term_id, _s_term_id]];
		ELSE
			_bigram_ids = _bigram_ids || ARRAY[_f_term_id, _s_term_id];
		END IF;
	END LOOP;
	
	FOREACH _bigram_id SLICE 1 IN ARRAY _bigram_ids LOOP
		SELECT g.gram_id INTO _gram_id
		FROM gram g
		WHERE g.term_id_1 = _bigram_id[1] AND g.term_id_2 = _bigram_id[2];

		IF _gram_id IS NULL THEN
			INSERT INTO gram(term_id_1, term_id_2) VALUES(_bigram_id[1], _bigram_id[2]) RETURNING gram_id INTO _gram_id;
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
