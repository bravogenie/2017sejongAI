#패키지 호출
from nltk.stem import WordNetLemmatizer 
#사용할 단어 입력
input_words = ['writings', 'organs', 'are', 'imagination', 'interesting', 'provisional', 'retarded', 'shrimps', 'cities', 'keeping', 'chocolate', 'bags'] 
#표제화(Lemmatizer) 객체 생성
lemmatizer = WordNetLemmatizer() 
#출력을 위한 문자열 포맷 정의
lemmatizer_names = ['NOUN LEMMATIZER', 'VERB LEMMATIZER']
formatted_text = '{:>24}' * (len(lemmatizer_names) + 1)
print('\n', formatted_text.format('INPUT WORD', *lemmatizer_names), '\n', '='*75) 
#단어별 명사, 동사 표제화 적용
for word in input_words:
    output = [word, lemmatizer.lemmatize(word, pos='n'),lemmatizer.lemmatize(word, pos='v')]
    print(formatted_text.format(*output)) 
