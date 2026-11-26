climber 패키지 파일 안내
========================

볼더링 강화학습 프로젝트의 기준 에이전트(연한 주황 마네킹 클라이머)입니다.
기준 체형: 키 1.75m / 몸무게 70kg / 양팔 너비 1.75m / 관절 25개 (모든 관절에 토크 모터)


[climber/ 폴더]

profile.py
  신체 사양서(BodyProfile). 키·팔다리 길이·어깨/골반 폭·몸무게·유연성(rom_scale)·근력(strength_scale)·그립.
  실험에서 신체 조건을 바꿀 때 주로 수정하는 파일.
  미리 만든 체형: baseline(기본), short_limbs(팔다리 -5%), long_limbs(팔다리 +5%)

scene.py
  벽과 홀드 배치(= 볼더링 문제, map).
  지금 default_wall()은 팀원 프로토타입과 같은 홀드 8개 배치 (175cm 몸에게는 좁아서 교체 예정).

model.py
  사양서(profile)와 벽(scene)을 받아 MuJoCo 모델을 만드는 생성기. 가장 중요한 파일.
  - 관절 사양표(JOINTS): 회전축, 최소/최대각, 감쇠, 최대 토크
  - 부위별 질량 비율(MASS_FRACTION): 실제 성인 남성 통계 (de Leva 1996)
  - 손 그립: 손-홀드를 붙이는 제약(grasp_왼오른_홀드이름)을 켜면 "잡기"
  보통은 build_model(profile) 하나만 호출하면 됨.

poses.py
  자세 계산 (물리 없이 기하 계산만).
  - t_pose(): 바닥에 서서 팔 벌린 자세
  - climb_pose(): 벽에 양손·양발로 매달린 자세
  렌더링용이고, 나중에 강화학습 시작 자세(reset)로 쓸 예정.
  주의: 충돌을 무시하므로 무릎 등이 벽 안에 들어갈 수 있음.

measure.py
  만든 캐릭터의 키·어깨 높이·고관절 높이·양팔 너비·질량을 실제로 측정.
  실험 결과 정리 시 체형 수치의 근거로 사용.

__init__.py
  패키지 입구. 예) from climber import BASELINE, build_model, climb_pose, body_stats


[다른 폴더]

scripts/view.py    마우스로 돌려 보는 MuJoCo 뷰어
                   python scripts/view.py
                   python scripts/view.py --pose t --profile long_limbs
                   python scripts/view.py --simulate     (물리 켜기)

scripts/render.py  사진(PNG) 만들기 → renders/ 폴더 (Git 제외)
                   python scripts/render.py

tests/test_climber.py
                   기준 사양을 지키는지 확인하는 테스트 10개. 코드 수정 후 꼭 실행.
                   python -m unittest discover tests


[아직 남은 일]
1. 손 그립 개선 (지금은 손이 홀드를 약 1.4cm 파고듦, 그립 힘 한계 미적용)
2. 시작 자세가 벽을 뚫지 않게 하기
3. 볼더링 문제(map) 하나 정하기
4. 강화학습 환경(관찰값·행동·보상) 만들기 + PPO 학습
