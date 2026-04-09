주제 : Minecraft VLA 프로젝트
목표 : Qwen3.5-9B + QLoRA + CraftJarvis/minecraft-vla-sft 데이터셋으로 마인크래프트를 플레이하는 VLA 모델 제작
사용 데이터셋 : CraftJarvis/minecraft-vla-sft


해야 할 것:
1. Qwen2-VL 액션 토큰 → Qwen3.5 토크나이저 호환 작업
- 방법 : 두 토크나이저의 어휘 사전 비교, 토큰 문자열 기준 재매핑(데이터셋 ID를 Qwen3.5 기준으로 변환하는 매핑 테이블 생성), 모델 임베딩 확장
2. vla 모델 만들기
3. vla 모델을 마크 서버에 접속하여 테스트(마크 서버는 이미 만들어져 있음)

해야할 것 1단계씩 할때마다 멈추고 나한테 알려주고 여기 환경은 실제 환경이 아니라서 코드 에러 없는지랑 dry run만 해줘