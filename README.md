# CPU 파인튜닝

이 프로젝트는 CPU 환경에서도 효과적으로 파인튜닝을 할 수 있도록 개발된 코드를 제공합니다. 

## 시작하기 전에

본 코드는 Python 3.10 버전에서 테스트하였습니다. 필요한 모든 의존성은 `requirements.txt` 파일에 명시되어 있습니다.

### 필요한 라이브러리 설치하기

필요한 모든 라이브러리를 설치하기 위해서는 아래 명령어를 실행해주세요:

```bash
pip install -r requirements.txt
```

### 참조
이 프로젝트는 Eduardo Alvarez의 블로그 포스트를 참조하였습니다. 링크는 아래 확인 부탁드립니다.<br/>
https://eduand-alvarez.medium.com/fine-tune-falcon-7-billion-on-xeon-cpus-with-hugging-face-and-oneapi-a25e10803a53

### 사용방법

다음 명령어로 파인튜닝 프로세스를 시작할 수 있습니다:

```bash
python trainning.py --use_ipex True --max_seq_length 512
```

