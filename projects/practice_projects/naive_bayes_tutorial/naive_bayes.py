import pandas as pd

def prepare_data_sklearn ():
	"Read dataset and preprocess via sklearn"

	# Tokenize and count
	from sklearn.feature_extraction.text import CountVectorizer
	
	# Split into training and test data
	from sklearn.cross_validation import train_test_split
	
	global df
	
	# Read dataset
	df = pd.read_table('smsspamcollection/SMSSpamCollection', sep='\t', header=None, names=['label','sms_message'])

	# Convert label entries to numerical values; 1 for spam and 0 for ham
	df.label = df.label.map({'ham':0, 'spam':1})

	# Split into training and testing sets
	global train_sms_message, test_sms_message, train_label, test_label
	train_sms_message, test_sms_message, train_label, test_label = train_test_split(
		df['sms_message'],
		df['label'],
		random_state=1)	

	# Learn training message vocabulary and convert to term-document matrix
	global training_data, testing_data
	count_vector = CountVectorizer()
	training_data = count_vector.fit_transform(train_sms_message);
	
	# Learn test message vocabulary
	testing_data = count_vector.transform(test_sms_message);
	
	return

def train_models ():
	"Fit training data to a MultinomialNB classifier"

	global bernoulli_nb, gaussian_nb, multinomial_nb
	
	# Multinomial Naive Bayes model
	from sklearn.naive_bayes import MultinomialNB 
	multinomial_nb = MultinomialNB()
	multinomial_nb.fit(training_data, train_label)

	# Gaussian Naive Bayes model
	from sklearn.naive_bayes import GaussianNB
	gaussian_nb = GaussianNB()
	gaussian_nb.fit(training_data.todense(), train_label)

	# Bernoulli Naive Bayes model
	from sklearn.naive_bayes import BernoulliNB
	bernoulli_nb = BernoulliNB()
	bernoulli_nb.fit(training_data, train_label)

def evaluate_models ():
	"Evaluate all models' performances"

	def evaluate_model (model, requires_dense=False):
		"Evaluate an individual model's performance"

		if (requires_dense == True):
			predictions = model.predict(testing_data.todense())
		else:
			predictions = model.predict(testing_data)

		#ACCURACY: Ratio of correct predictions to total number of predictions
		from sklearn.metrics import accuracy_score
		print('Accuracy score:   ', format(accuracy_score(test_label, predictions)))

		#PRECISION: Ratio of true positives to predicted positives
		from sklearn.metrics import precision_score
		print('Precision score:  ', format(precision_score(test_label, predictions)))

		#RECALL AKA SENSITIVITY: Ratio of true positives to number of actual spam
		from sklearn.metrics import recall_score
		print('Recall scores:    ', format(recall_score(test_label, predictions)))

		#F1 SCORE: Weighted average of precision and recall. Ranges from 0 to 1 with 1 being the best possible
		from sklearn.metrics import f1_score
		print('F1 score:         ', format(f1_score(test_label, predictions)))	

		#BRIER SCORE LOSS: Mean squared difference between predicted probablility (0 or 1 in our case) and actual outcome
		from sklearn.metrics import brier_score_loss
		print('Brier score loss: ', format(brier_score_loss(test_label, predictions)))

		print("");

		return

	print("Multinomial NB")
	evaluate_model(multinomial_nb)

	print("Bernoulli NB")
	evaluate_model(bernoulli_nb)

	print("Gaussian NB")
	evaluate_model(gaussian_nb, True)
	
	return

# Read and Preprocess data set, then divide into split into Training and Test sets
prepare_data_sklearn()

# Train a Multinomial Naive Bayes model using the test data
train_models()

# Evaluate the models' predictions
evaluate_models()
