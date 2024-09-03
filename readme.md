# SNS 사진 분석 댓글 및 피드백 프로젝트

- 파이썬 버전
    ```bash
    conda create -n project_3 python==3.11
    ```
- 패키지 설치
    ```bash
    git clone https://github.com/crazy2894/project_3_git.git
    cd project_3_git
    pip install -r requiremets.txt
    ```
## Language Model

### gpt 또는 gemini 를 이용한 데이터 셋 생성
- 사용 데이터 셋 : [gpt 생성 데이터](data/text_data/output_text.json)
  - [gpt 생성 코드](code_data_gen/3_textdata_generating.ipynb)
  - 프롬프팅 : 
    ```py
    def prompting(input_):
    return f"""
    사진에 대한 댓글 입력
    질문 금지
    sns 사진 요소 : '분노, 여자, 인간의 얼굴, 공, 의류'
    sns 댓글 : 화가 난 듯한 표정이네! 🏀 공은 무슨 종류야? 옷도 멋지다! 😊
    
    예측 한 문장
    sns 사진 요소 : {input_}
    sns 댓글 : """
    ```
  - 출력 예시 :
    ```py
    input_ = '공포, 헤드폰'
    output_ = '아찔한 분위기네요! 🎧 어떤 음악 듣고 계신가요? 궁금해요! 😊'
    ```

### t5 (Text-to-Text Transfer Transformer)

- 학습 데이터 형식
  ```
  input_data = ['슬픔, 분노', ...]
  output_data = ['감정이 복잡해 보이네요… 힘든 날이신가요? ❤️', ... ]
  ```

#### 모델 훈련
  - transfer_0 : 기본값으로 훈련
  - transfer_1 : 
    - 드롭아웃 비율 0.1 -> 0.2
    - 훈련시 추가 인자
      ```
      learning_rate=5e-5,          # 기본값에서 시작
      lr_scheduler_type="linear",  # 스케줄러
      warmup_steps=500,            # 500 스텝 동안 학습률을 점진적으로 증가
      weight_decay=0.01,           # l2 정규화 기법 중 하나
      max_grad_norm=1.0,           # 그라디언트 클리핑
      ```
    - transfer_1 : 로컬 환경 및 기본 base 모델 이용
    - transfer_1_large_colab : colab 환경 및 large 모델이용

  - t5 비교 그래프
    ![비교 그래프](models/t5/val_loss_comparison.png)
  
  - 결론 : 세 모델의 큰 차이는 없어 보인다. 그러므로 이중 효율좋고 loss 최저값이 낮은 1번(default) 로 선택 
    - 각 모델별 특징
    ```text
    # loss 최저
    - default = 0.1783
    - setting 1 = 0.1790
    - setting 1 with colab = 0.1745

    # 요구 vram
    - default = 6gb
    - setting 1 = 6gb
    - setting 1 with colab = 29gb
    ```




### gpt2 (*Language Models are* **Unsupervised** *Multitask Learners*)
- 즉 정답 라벨은 없다. (비지도 학습)
  - 학습 데이터 형식
    ```
    input_data = ['슬픔, 분노, 감정이 복잡해 보이네요… 힘든 날이신가요? ❤️', ... ]
    ```
    또는 텍스트 로 시퀀스 데이터 안에 명시한다.
    ```
    input_data = ['입력 : 슬픔, 분노 \n 출력 : 감정이 복잡해 보이네요… 힘든 날이신가요? ❤️', ... ]
    ```
    또한 예측시 모델의 입력값으로 ```'입력 : 슬픔, 분노 \n 출력 : ``` 와 같이 입력하여 출력값을 얻어야함

- gpt2_base_0 : gpt2 베이스 모델 이용
- kogpt2_0 : 기본 하이퍼 파라미터 이용
- kogpt2_1 : 
    - 훈련시 추가 인자
      ```
      learning_rate=5e-5,          # 기본값에서 시작
      lr_scheduler_type="linear",  # 스케줄러
      warmup_steps=500,            # 500 스텝 동안 학습률을 점진적으로 증가
      weight_decay=0.01,           # l2 정규화 기법 중 하나
      max_grad_norm=1.0,           # 그라디언트 클리핑
      ```

<details>
  <summary>삭제 내용</summary>
  
  ### 2024-09-02
    code\1_데이터_확인.ipynb  : fix
    requiremets.txt         : 필요한 라이브러로 수정(업데이트 중)
  ## 파일 구조

  ### 📁 code : 모델 훈련 및 예측, 데이터 확인관련 코드
  ```text
  1_데이터 확인.ipynb           # 데이터 확인 코드
  2_od_YOLO_finetunning.ipynb  # wassup 얼굴 데이터 전이학습
  2_od_YOLO_lvis.ipynb         # lvis 데이터셋 전이학습
  3_lm_gpt2finetunning         # gpt2 전이학습
  3_lm_t5                      # t5 전이학습
  4_pipe_line                  # 입력단부터 최종 출력단 까지으 파이프라인
  ```
  
  ### 모델 설명
  ```text
  yolov8m-oiv7.pt              # 객체 검출 모델 중간 사이즈
  yolov8x-oiv7.pt              # 객체 검출 모델 라지 사이즈
  yolov10n-face.pt             # wassup dataset으로 전이학습한 모델
  ```
  
  ### 📁 code_data_gen : api 를 이용한 코드
  ```text
  1_chat_gpt_translate.ipynb   # 텍스트 번역 모델 (oiv7 의 정답 라벨 번역을 위한 코드)
  2_img_pred_and_gen.ipynb     # 이미지를 모델 입력 후 출력 값을 결과를 저장하는 코드
  3_textdata_generating.ipynb  # text to text 로 댓글 데이터 생성 코드
  ```
</details>

# 링크 : [진행과정 표](https://docs.google.com/spreadsheets/d/1OklwBcfJiqlj7JJHE1Pez9jpgLctun0BPKrBD4HW2A0/edit?gid=1967477975#gid=1967477975) , [기획안](https://docs.google.com/presentation/d/1HKMJk6zLfsEqedcVdcQipHY8V8snd6oP2ajS9FDFgKI/edit#slide=id.p), 