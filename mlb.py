import requests
import json
import base64
import binascii 
import binhex
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.grid_search import GridSearchCV

session = requests.Session()

#r = session.get("https://mlb.praetorian.com/challenge")
#print r.text
#r = session.post("https://mlb.praetorian.com/solve", data=json.dumps({"target":"arm"}))
#print r.text

corpus = {'avr':[], 'alphaev56':[], 'arm':[], 'm68k':[], 'mips':[], 'mipsel':[], 'powerpc':[], 's390':[], 'sh4':[], 'sparc':[], 'x86_64':[], 'xtensa':[]}


def init_db():
    c = db.cursor()
    c.execute('''
        create table data(target TEXT PRIMARY KEY, label TEXT)
    ''')
    db.commit()   
 

def check_min():
    c = db.cursor()
    for isa in corpus.keys():
        c.execute("SELECT COUNT(target) FROM data WHERE label LIKE '" + isa + "'")
        row = c.fetchone()
        if row[0] < 10000:
            return False
    return True
        
#   for key, value in corpus.items():
#        if len(value) < 10:
#            return False
#    return True

def splitCount(s, count):
     return [''.join(x) for x in zip(*[list(s[z::count]) for z in range(count)])]

def init_corpus():
    while not check_min():
        challenge = session.get("https://mlb.praetorian.com/challenge")
        challenge_response = challenge.json()
	#print challenge_response	

        solve = session.post("https://mlb.praetorian.com/solve", data=json.dumps({"target":"arm"}))
        solve_response = solve.json()
        #print solve_response

	isa =  solve_response['target'].decode('ascii') 
	binary = challenge_response['binary'].decode('ascii')
	binary = binascii.hexlify(base64.b64decode(binary))

	#binary = ' '.join(splitCount(binary, 2))

#	print isa + " ==> " + binary
	#corpus[isa].append(binary)

        try:        
            cursor = db.cursor()
            cursor.execute('''INSERT INTO data(target, label) VALUES(?,?)''', (binary, isa))
            db.commit()
        except:
            pass

    #print corpus

db = sqlite3.connect('training_set.db')

#init_db()

init_corpus()

#for key in corpus:
#    print key + " " + str(len(corpus[key]))

"""
print "[+] sqlite results:"
c = db.cursor()
for isa in corpus.keys():
    c.execute("SELECT COUNT(target) FROM data WHERE label LIKE '" + isa +"'")
    row = c.fetchone()
    print "[+] \t" + str(row[0]) + " results for " + isa + " architecture"
    c.execute("SELECT target FROM data WHERE label LIKE '" + isa + "'")
"""

hex_train = []
target_train = []

c = db.cursor()
for key in corpus.keys():
    c.execute("SELECT target FROM data WHERE label LIKE '" + key + "'")
    rows = c.fetchall()
    for row in rows:
        hex_train.append(str(row[0]))
        target_train.append(key)

vec_opts = {
    "ngram_range": (2,3),
    "analyzer": "word",
    "token_pattern": "..",
    "min_df": 0.35,
}

v = CountVectorizer(**vec_opts)
X = v.fit_transform(hex_train, target_train)

idf_opts = {"use_idf": True}
idf = TfidfTransformer(**idf_opts)

X = idf.fit_transform(X)

clf = MultinomialNB().fit(X, target_train)
parameters = {
    'ngram_range': [(1,1),(1,2)],
    'min_df': (0,1),
    'use_idf': (True,False),
}
gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X, target_train)


for x in range(0, 1):
	test = session.get("https://mlb.praetorian.com/challenge")
	test_response = test.json()
	# print test.json()
	binary = test_response['binary'].decode('ascii')
	binary = binascii.hexlify(base64.b64decode(binary))

	temp = [binary]
	X_new_counts = v.transform(temp)
	X_new_tfidf = idf.transform(X_new_counts)

	#predicted = clf.predict(X_new_tfidf)

	#print str(predicted) + " ==> " + binary

	predicted = gs_clf.predict(temp)

	best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
	for param_name in sorted(parameters.keys()):
	    print param_name + ": " + best_parameters[param_name]


	solve = session.post("https://mlb.praetorian.com/solve", data=json.dumps({"target":str(predicted[0])}))
	solve_response = solve.json()
#	print solve.json()
	solve_response = json.loads(json.dumps(solve_response))
	if str(predicted[0]) != solve_response['target']:
            print "[!] ",
        else:
            print "[+] ",
	print "guess is: " + str(predicted[0]) + "\tactual: " + solve_response['target'] + " correct: " + str(solve_response['correct']) + " accuracy: " + str(solve_response['accuracy'])

db.close()

