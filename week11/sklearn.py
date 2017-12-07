
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#카테고리 맵 정의
category_map = {'talk.politics.misc': 'Politics', 'rec.autos': 'Autos',
		'rec.sport.hockey': 'Hockey', 'sci.electronics': 'Electronics',
		'sci.med': 'Medicine'}

#학습데이터셋을 받음
training_data = fetch_20newsgroups(subset='train',
				   categories=category_map.keys(), shuffle=True, random_state=5) 
# countvectorizer 객체를 사용해 단어 빈도를 추출
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(training_data.data)
print("\nDimensions of training data:", train_tc.shape)

# tf-idf 변환기 생성
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc) 
# 테스트 데이터 정의
input_data = [     'You need to be careful with cars when you are driving on slippery roads',      'A lot of devices can be operated wirelessly',     'Players need to be careful when they are close to goal posts',     'Political debates help us understand the perspectives of both sides' ] 
# 다항 분포 나이브 베이즈 분류기 학습
classifier = MultinomialNB().fit(train_tfidf, training_data.target) 
# count vectorizer를 사용해 입력 데이터 변환
input_tc = count_vectorizer.transform(input_data) 
# tfidf 변환기를 사용해 벡터 데이터 변환 
input_tfidf = tfidf.transform(input_tc) 
# 카테고리 예측
predictions = classifier.predict(input_tfidf) 
# 결과 출력
for sent, category in zip(input_data, predictions):
	print('\nInput:', sent, '\nPredicted category:',\
	      category_map[training_data.target_names[category]]) 
