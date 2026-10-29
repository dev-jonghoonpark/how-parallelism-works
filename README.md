# How Parallelism Works

**모델 하나를 GPU 만 대에 나눠 싣는 법.** 초거대 모델 학습에 쓰는 병렬화 방식들(데이터 병렬과 ZeRO/FSDP, Megatron식 텐서 병렬, Mesh-TensorFlow 레이아웃, GPipe 계열 파이프라인, 컨텍스트 병렬, 전문가 병렬)이 각각 무엇을 자르고 어떤 통신을 치르는지 브라우저에서 계산하고 그려 보는 인터랙티브 교육 자료입니다. 마지막에는 Meta가 Llama 3 405B를 16,384개 H100에 TP 8 × CP × PP 16 × DP로 어떻게 배치했는지 그림으로 따라갑니다.

🔗 **[데모 바로가기](https://dev-jonghoonpark.github.io/how-parallelism-works/)**

## 다루는 내용

1. **왜 잘라야 하나** · 혼합 정밀도 + Adam의 파라미터당 16바이트, 405B → 6.5TB
2. **네 논문, 네 가지 자르기** · GPipe(파이프라인), Megatron-LM(텐서), Mesh-TensorFlow(레이아웃), ZeRO(데이터 병렬 개선). "순수 파이프라인은 최후 수단"이 가리키는 것
3. **데이터 병렬과 ZeRO / FSDP** · 단계별 장치당 메모리 계산기(논문 예 120GB → 31.4 / 16.6 / 1.9GB 재현), GPU별 보유 조각 그림
4. **텐서 병렬** · Megatron MLP의 열·행 분할 그림, 층 하나의 계산 대 all-reduce 시간을 NVLink와 서버 사이 연결로 비교
5. **Mesh-TensorFlow** · 같은 텐서를 네 가지 레이아웃으로 2 × 4 메시에 놓아 보기
6. **파이프라인 병렬** · 버블 (p−1)/(v·m+p−1), interleaved 스케줄, 최후 수단인 이유
7. **컨텍스트 병렬** · Ring Attention과 all-gather, 인과 마스크 부하 균형(2·CP 조각 짝짓기)
8. **전문가 병렬** · all-to-all 라우팅 그림, DeepSeek-V3 구성
9. **Llama 3 405B의 4D 병렬화** · 계층 그림, 논문 그림 5(16 GPU) 그룹 탐색기, 표 4 세 구성을 노드 단위로 그린 지도, 파이프라인·FSDP·CP 세부 조정
10. **정리** · 축을 고르는 순서
11. **퀴즈** 7문항

## 참고

- Llama Team, AI @ Meta (2024), *The Llama 3 Herd of Models* — [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- Huang et al. (2019), *GPipe* — [arXiv:1811.06965](https://arxiv.org/abs/1811.06965)
- Shoeybi et al. (2019), *Megatron-LM* — [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Narayanan et al. (2021), *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* — [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- Shazeer et al. (2018), *Mesh-TensorFlow* — [arXiv:1811.02084](https://arxiv.org/abs/1811.02084)
- Rajbhandari et al. (2020), *ZeRO* — [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- Zhao et al. (2023), *PyTorch FSDP* — [arXiv:2304.11277](https://arxiv.org/abs/2304.11277)

[how-ai-works](https://github.com/dev-jonghoonpark/how-ai-works) 시리즈의 일부입니다.
