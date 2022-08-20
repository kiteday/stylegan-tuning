# stylegan-tuning
스타일겐이랑 s2fgan 튜닝한 코드 모아논 얼렁뚱땅 리포
정리안됨 주의

S2FGAN-master.zip : 과거 튜닝코드 최신 X

## S2FGAN
### Add
* generater.py
* train.py에 checkpoint부터 학습 추가

### tuning version 1 : 스케치 없이 grund truth이미지로만 학습
* tuning_dataset.py
* tuning_model.py
* tuning_train.py
* tuning_generater.py

### tuning version2 : ver1에서 label 제거
* tun_model2.py
* tun_train2.py
* tun_dset2.py : 기존 버전이 오류가 나서 새로 만듦
* tun_generate2.py : 샘플이미지를 주어주고 거기서 이미지 생성

### tuning version3 : 모델 학습 시, g_ema 추가 저장
* tun_model3.py
* tun_train3.py 

* tun_generate3.py : 2와는 달리 모델을 가져오지 않고 모델에 저장한 g_ema만 가져옴
