from  nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer
input_text = "So, here we are, poised to begin round six of Brexit talks, and it might move you to raise an eyebrow or two to hear that the two sides can't even agree what to call these meetings now: negotiating rounds, stock-taking exercises or an information exchange. These days I'd call them a dance around a standstill. UK negotiators have long been frustrated with the format of the (so far) monthly rounds of talks. They feel that sitting in Brussels for days at a stretch does not allow them the opportunity to consult London when an impasse is reached in order to - maybe - come up with a plan B and - hopefully - move forward."
print("\nSentence Tokenizer:")
print(sent_tokenize(input_text))
print("\nWord tokenizer : ")
print(word_tokenize(input_text)) 
print("\nWord Punct Tokenizer : ")
print(WordPunctTokenizer().tokenize(input_text))

from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

#사용할 단어 입력
input_words = ['colonization', 'social', 'are', 'branded', 'coward', 'randomize', 'possibly', 'proverbs', 'hospital', 'keep', 'scratchy', 'banana']
#스테머 객체 생성 
porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer('english') 
#출력을 위한 문자열 포맷 정의 
stemmer_names = ['PORTER', 'LANCASTER', 'SNOWBALL']
formatted_text = '{:>16}' * (len(stemmer_names)+1)
print('\n', formatted_text.format('INPUT WORD', *stemmer_names), '\n', '='*68) 
#입력 단어별로 어간 추출해 출력
for word in input_words:
    output=[word, porter.stem(word), lancaster.stem(word), snowball.stem(word)]
    print(formatted_text.format(*output)) 
