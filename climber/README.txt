climber 패키지 파일 안내
========================

볼더링 강화학습 프로젝트의 기준 에이전트(연한 주황 마네킹 클라이머)입니다.
기준 체형: 키 1.75m / 몸무게 70kg / 양팔 너비 1.75m / 관절 27개 (모든 관절에 토크 모터)


[climber/ 폴더]

profile.py
  신체 사양서(BodyProfile). 키·팔다리 길이·어깨/골반 폭·몸무게·유연성(rom_scale)·근력(strength_scale)·그립.
  실험에서 신체 조건을 바꿀 때 주로 수정하는 파일.
  미리 만든 체형: baseline(기본), short_limbs(팔다리 -5%), long_limbs(팔다리 +5%)

scene.py
  벽과 홀드 배치(= 볼더링 문제, map).
  default_wall(): 팀원 프로토타입과 같은 홀드 8개 배치 (175cm 몸에게는 좁음)
  test_wall()   : 175cm 몸에 맞춘 홀드 4개 (테스트·뷰어·렌더에서 사용). 볼더링 문제가 정해지면 교체.

model.py
  사양서(profile)와 벽(scene)을 받아 MuJoCo 모델을 만드는 생성기. 가장 중요한 파일.
  - 관절 사양표(JOINTS): 회전축, 최소/최대각, 감쇠, 최대 토크
  - 부위별 질량 비율(MASS_FRACTION): 실제 성인 남성 통계 (de Leva 1996)
  - 손 그립: 손-홀드를 붙이는 제약(grasp_왼오른_홀드이름)을 켜면 "잡기"
  보통은 build_model(profile) 하나만 호출하면 됨.

contact.py
  손·발 접촉 규칙. 매 스텝 ContactModel.update(data) 호출.
  - 손 잡기: 손바닥이 홀드에 닿고, 손바닥 방향과 홀드 법선이 20° 이내로 마주 볼 때
    → 실제 접촉점에 손을 고정 (손이 홀드를 파고들지 않음)
  - 손 놓치기: 손이 홀드에 거는 힘이 그 홀드가 버틸 수 있는 방향(pull_dir ± pull_cone_deg)을 벗어나거나,
    그립 한계(grip_capacity)를 넘을 때. 놓친 손은 0.3초간 다시 못 잡음
    (홀드 종류: 저그=아래로 당기기(기본), 사이드풀=옆, 언더클링=위. scene.py의 Hold 참고)
  - 악력 피로: 잡고 있으면 줄고(악력 한계로 매달리면 60초마다 절반, 가볍게 매달리면 더 느리게),
    놓고 쉬면 회복(30초면 95%), 처음 악력의 30% 밑으로는 안 내려감. 값은 profile.py에 있음
  - 발: 고정하지 않음(누르기만 가능). '지지 중'인지만 판정 (관찰값·보상용)

control.py
  PoseHold: 시작 자세를 유지하는 가장 단순한 PD 제어기. 이걸로 스스로 매달려 버틴다 (이동은 못 함).
  팔은 약하게, 몸통·다리는 강하게. 나중에 강화학습과 비교할 기준선.

poses.py
  자세 계산 (물리 없이 기하 계산만).
  - t_pose(): 바닥에 서서 팔 벌린 자세
  - climb_pose(): 벽에 양손·양발로 매달린 자세 (손바닥은 홀드 앞면에서 벽 쪽을 향하게, 발바닥은 평평하게)
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
                   python scripts/view.py --pose t --profile long_limbs (기존 T-Base pose)
                   python scripts/view.py --mode hang    (물리 켬 + PoseHold: 스스로 매달려 버팀)
                   python scripts/view.py --mode assist  (hang + 골반 보조: 마우스로 손을 당겨 놓치는 것 보기)
                   python scripts/view.py --mode fall    (물리 켬 + 힘 0: 아직 버티는 제어기가 없어 떨어짐)

scripts/render.py  사진(PNG) 만들기 → renders/ 폴더 (실험 기록과 함께 Git에 올림)

scripts/hang_test.py  악력이 떨어져 손을 놓칠 때까지 매달리기 (창 없이 빠르게), 악력·하중 기록 → renders/*.csv
                   python scripts/hang_test.py --grip 500
scripts/plot_hang.py  위 기록을 그래프로 → renders/hang_fatigue.png

scripts/compare_hang.py  클라이머 3명(악력 850/500/350N)을 나란히 매달고, 각자 위에 악력·하중·피로도 표시
                   python scripts/compare_hang.py          (4배속 영상 → renders/compare_hang.mp4)
                   python scripts/compare_hang.py --live   (창으로 바로 보기, q로 종료)
                   python scripts/compare_hang.py --grips 900,600,400   (악력 바꾸기)
                   python scripts/render.py

tests/test_climber.py, tests/test_contact.py
                   기준 사양·손발 접촉 규칙·스스로 매달리기·악력 피로를 확인하는 테스트 24개. 코드 수정 후 꼭 실행.
                   python -m unittest discover tests


[아직 남은 일]
1. 발 지지 개선: 지금은 매달려 있는 동안 발이 발 홀드에서 미끄러져 내려간다 (손으로만 버팀)
2. 악력 피로 값(60초/30초/30%)을 실제 클라이밍 자료로 조정
3. 볼더링 문제(map) 하나 정하기
4. 강화학습 환경(관찰값·행동·보상) 만들기 + PPO 학습
