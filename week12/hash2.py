from PIL import Image
import numpy as np

# 이미지 데이터를 Averages Hash로 변환 함수 선언
def average_hash(fname, size = 16): #average_hash(파일이름, 사이즈)
      img = Image.open(fname) # 이미지 데이터 열기
      img = img.convert('L')  # 그레이스케일로 변환
                              #'1'지정하게 되면 이진화 그밖에 "RGB", "RGBA", "CMYAK" 모드 지정가능
      img = img.resize((size, size), Image.ANTIALIAS) # 리사이즈
      pixel_data = img.getdata() # 픽셀 데이테 가져오기
      pixels = np.array(pixel_data) # Numpy 배열로 변환하기
      pixels = pixels.reshape((size, size)) # 2차원 배열로 변환
      avg = pixels.mean() # 평균 구하기
      diff = 1*(pixels>avg) #평균보다 크면 1, 작으면 0으로 변환하기
      return diff


# 이전 해시로 변환하는 함수 선언
def np2hash(n):
      bhash = []
      for n1 in ahash.tolist():
            s1 = [str(i) for i in n1]
            s2 = "".join(s1)
            i = int(s2,2) #이진수를 정수로 변환하기
            bhash.append("%04x"%i)
      return "".join(bhash)

# Average Hash 출력하기
avhash_1 = average_hash('genie.jpg')
avhash_2 = average_hash('genie.jpg')
a = avhash_1.reshape(1,-1) # 2차원 배열이 다시 1차원 배열로 변환됨
b = avhash_2.reshape(1,-1) # 2차원 배열이 다시 1차원 배열로 변환됨
print((a == b)) # a와 b의 같은부분 다른 부분 출력
print('\n',(a != b).sum()) # 다른 포인트 출력 = 해밍 거리
