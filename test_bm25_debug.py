from bm25_index import BM25Retriever

r = BM25Retriever()
corpus = [{'chunk_id':'1', 'text':'Hello world'}, {'chunk_id':'2', 'text':'Hello universe'}, {'chunk_id':'3', 'text':'Another completely different text'}]
r.build_bm25_index(corpus)
print("Scores for 'world':", r.bm25.get_scores(['world']))
print("Search results for 'world':", r.bm25_search('world'))
