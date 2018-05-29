import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
from pprint import pprint
import psycopg2
import pymongo
from bson.objectid import ObjectId
import re
from sklearn.feature_extraction.text import CountVectorizer
import sys
import time
from SimpleCache import SimpleCache

DEFAULT_QUERY_WEIGHTS = {
    'fulltext': 0.4,
    'title': 0.2,
    'abstract': 0.2,
    'authors': 0.2,
}

class PsqlTfIdfDocSearchEngine:
    pg_conn = None
    stemmer = None
    mongo_db = None
    wiki_collection = None
    
    def __init__(self):
        self.pg_conn = psycopg2.connect("dbname='test_sharesci' user='sharesci' host='127.0.0.1' password='sharesci'")
        mongo_client = pymongo.MongoClient('127.0.0.1', 27017)

        self.mongo_db = mongo_client['sharesci']
        self.wiki_collection = self.mongo_db['wiki']

        self.stemmer = PorterStemmer(mode=PorterStemmer.MARTIN_EXTENSIONS)
        self._cache = SimpleCache()


    def search_qs(self, query, max_results=sys.maxsize, offset=0, getFullDocs=False):
        if query is None or not re.match(r'[ \w]*\w[ \w]*', query):
            print('fail1')
            return
        query = re.sub(r"\s+", " ", query.lower())
    
        cache_extra_keys = {'max_results': max_results, 'offset': offset}
        main_key = 'search_qs.' + query
        cached_result = self._cache.get(main_key, cache_extra_keys)
        if cached_result is not None:
            return cached_result

        query_vec = self.make_query_vector(query)

        term_idfs = self.get_idfs([v[0] for v in query_vec])

        #if print_idfs:
        #    print("IDF values for terms: ", term_idfs)

        query_tuples = []
        query_l2 = 0.0
        for qterm in query_vec:
            term = qterm[0]
            raw_count = qterm[1]
            tf = 1 + np.log(raw_count) if raw_count != 0 else 0
            tfidf = tf * term_idfs[term][1]
            query_tuples.append([term_idfs[term][0], tfidf])
            query_l2 += tfidf * tfidf
        query_l2 = np.sqrt(query_l2)
        query_tuples = [(tup[0], tup[1] / query_l2) for tup in query_tuples]

        results = self.query_cosine_similarities(query_tuples, max_results=max_results, weights=DEFAULT_QUERY_WEIGHTS)

        self._cache.add(main_key, cache_extra_keys, results)

        return results

    ## Compute the most similar documents, given a query vector
    #
    # "Similarity" here refers to cosine similarity. This function considers all
    # documents, including the sub-documents representing titles, abstracts,
    # authors, etc, and computes the cosine similarity of each with the query
    # vector. It then takes a weighted sum of grouped documents for the final
    # similarity score. A document "group" consists of the main (fulltext)
    # document and all sub-documents (i.e., documents which are in a parent-child
    # relationship with the main document in the database).
    #
    # @param query_tfidf_tuples (list)
    # <br>	Format: `[(gram_id (int), term_ltc (float)), ...]`
    # <br>	-- a query vector containing tuples of gram IDs paired with their 
    # 	TF-IDF score
    #
    # @param max_results (int)
    # <br>	-- The maximum number of results to return
    #
    # @return (list)
    # <br>	Format: `[(text_id (str), similarity (float)), ...]`
    # <br>	-- A list of tuples containing document IDs paired with similarity 
    # 	scores, sorted by similarity.
    def query_cosine_similarities(self, query_tfidf_tuples, max_results=20, weights=DEFAULT_QUERY_WEIGHTS):
        cur = self.pg_conn.cursor()
        result = None
        try:
            values_str = ','.join(
                cur.mogrify('(%s, %s)', (tfidf_tuple[0], tfidf_tuple[1])).decode() for tfidf_tuple in query_tfidf_tuples)
        except psycopg2.Error as err:
            print('Failed to stringify values table for cosine similarity query', file=sys.stderr)
            print(err.diag.message_primary, file=sys.stderr)

        sql = """
            SELECT similarity, text_id 
            FROM (
            SELECT (SELECT text_id FROM document d2 WHERE d2._id = dg_id) AS "text_id", similarity 
            FROM (
                SELECT COALESCE(document.parent_doc, document._id) AS "dg_id",
                    COALESCE(SUM(lnc*term_ltc*(
                        CASE document.type 
                            WHEN 1 THEN {fulltext_weight:0.4f} 
                            WHEN 2 THEN {title_weight:0.4f} 
                            WHEN 3 THEN {abstract_weight:0.4f} 
                            WHEN 4 THEN {authors_weight:0.4f} 
                            WHEN 5 THEN {fulltext_weight:0.4f} 
                            ELSE 0.0 
                        END)
                    ), 0) AS similarity
                FROM tf
                INNER JOIN (VALUES {valuetbl}) AS query_matrix(query_gram_id, term_ltc)
                    ON CAST(query_gram_id AS INT)=tf.gram_id
                INNER JOIN gram
                    ON (gram.gram_id = tf.gram_id)
                RIGHT OUTER JOIN document
                    ON document._id=doc_id
                GROUP BY dg_id
                ORDER BY similarity DESC LIMIT %s
            ) AS subquery_1
            ) AS subquery_2
            ;
        """.format(
            valuetbl=values_str,
            fulltext_weight=weights['fulltext'],
            title_weight=weights['title'],
            abstract_weight=weights['abstract'],
            authors_weight=weights['authors'],
        );

        try:
            cur.execute(sql, (max_results,))
            result = cur.fetchall()
        except psycopg2.Error as err:
            print('Failed to get cosine similarities', file=sys.stderr)
            print(err.diag.message_primary, file=sys.stderr)
        self.pg_conn.commit()
        cur.close()
        return result

    ## Create query vector from a query string
    #
    # @param query_string (str)
    # <br>	-- The query string
    #
    # @return (list) a vector of query terms. Each term is a tuple of two strings,
    # 	representing a bigram (or unigram if the second term is '')
    #
    def make_query_vector(self, query_string):
        query_tokens = [self.stemmer.stem(token) for token in nltk.word_tokenize(query_string)]
        # query_tokens = [token for token in nltk.word_tokenize(query_string)]
        query_vec = []
        for i in range(len(query_tokens)):
            tok1 = query_tokens[i]
            tok2 = ''
            if i < len(query_tokens) - 1:
                query_vec.append(((tok1, ''), 1))
                tok2 = query_tokens[i + 1]
            query_vec.append(((tok1, tok2), 1))
        # for tok1 in query_tokens:
        #	query_vec.append(((tok1, ''), 1))
        #	for tok2 in query_tokens:
        #		query_vec.append(((tok1, tok2), 1))
        return query_vec;

    ## Get the IDF values for the given terms
    # 
    # @param terms (list-like)
    # <br>	Format: A list of terms (each term as str)
    # 
    # @return (dict) 
    # <br>	-- a dict with keys being terms (as str) and values being tuples
    # 	of `(gram_id, IDF)`
    def get_idfs(self, terms):
        cur = self.pg_conn.cursor()
        result = None
        num_docs = 1
        try:
            cur.execute("SELECT COUNT(*) from document")
            num_docs = int(cur.fetchone()[0])
            values_str = ','.join(cur.mogrify('(%s, %s)', (term[0], term[1])).decode() for term in terms)
            sql = """
                SELECT query_term1, query_term2, gram.gram_id, COALESCE(df, 1) 
                FROM (VALUES {}) AS tmp_query(query_term1, query_term2) 
                LEFT OUTER JOIN gram ON (
                        term_id_1 = (SELECT term._id FROM term WHERE term.term=query_term1) 
                        AND term_id_2 = (SELECT term._id FROM term WHERE term.term=query_term2)
                    )
                    LEFT OUTER JOIN idf ON (idf.gram_id = gram.gram_id);
                """.format(values_str);
            cur.execute(sql)
            result_tuples = cur.fetchall();
            result = {}
            for t in result_tuples:
                term = (t[0], t[1])
                gram_id = t[2]
                df = float(t[3]) + 1  # Add 1 to protect against div-by-zero
                idf = np.log(1 + (num_docs / df))
                result[term] = (gram_id, idf)
        except psycopg2.Error as err:
            print('Failed to get term DFs', file=sys.stderr)
            print(err.diag.message_primary, file=sys.stderr)
        self.pg_conn.commit()
        cur.close()
        return result






def pretty_print_metadata_results(results):
    print('{:>2s}  {:100s}  {:15s}  {:7s}        '.format('#', 'Title', 'wiki id', 'Score'))
    result_num = 1
    for result in results:
        print('{:2d}. {:100s}  {:15s}  {:0.5f}    '.format(result_num, re.sub('[ ]*\n[ ]*', ' ', result['title']),
                                                           result['id'], result['score']))
        result_num += 1



if __name__ == '__main__':

    query = None
    tfidf = None
    try:
        while query != 'exit':
            times = {}

            query = input('Type your query: ')
            
            tfidf = PsqlTfIdfDocSearchEngine()

            #start_time = time.perf_counter()
            doc_scores = tfidf.search_qs(query, max_results=20) 
            print(doc_scores)
            #times['query'] = time.perf_counter() - start_time

            #start_time = time.perf_counter()
            #metadata_results = attach_metadata(doc_scores)
            metadata_results = []
            for result in doc_scores:
                metadata = {'raw_id': result[1], 'title': '', 'id': '', 'score': result[0]}
                if len(result[1]) == 24:
                    mongo_result = tfidf.wiki_collection.find({'_id': ObjectId(result[1])})[0]
                    metadata['title'] = mongo_result['title'];
                    metadata['id'] = mongo_result['id'];
                    metadata['_id'] = result[1];
                else:
                    metadata['id'] = result[1];
                metadata_results.append(metadata);
            #times['mongo'] = time.perf_counter() - start_time

            print("The top 20 scores are:")
            pretty_print_metadata_results(metadata_results)
           # print('\n{:0.4f}s to perform the query, {:0.4f}s to get the metadata for results from Mongo\n'.format(
           #     times['query'], times['mongo']))
    except EOFError as err:
        print('exit')

    tfidf.pg_conn.close()
    tfidf.mongo_client.close()
