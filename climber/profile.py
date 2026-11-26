"""Body profile: the physical traits that differ between climbers.

[한국어 설명]
클라이머 한 명의 "신체 사양서"를 정의하는 파일.
키·팔다리 길이·몸무게·유연성·근력·그립 같은, 실험에서 바꿔 볼 신체 조건이 전부 여기 있다.
BodyProfile 값만 바꾸면 model.py가 그 체형에 맞는 캐릭터를 자동으로 다시 만든다.

사용 예:
    from dataclasses import replace
    heavy = replace(BASELINE, name="heavy", total_mass=85.0)   # 몸무게만 85kg인 체형
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class BodyProfile:
    """Segment lengths are metres; *_scale values are multipliers on the baseline body.

    The baseline is an adult male, 1.75 m (sole to head top) and 70 kg. Segment lengths
    follow Winter's height ratios (upper arm 0.186 H, forearm 0.146 H, thigh/shank ~0.245 H,
    shoulder height ~0.82 H, hip joint ~0.53 H). total_mass is split over the segments with
    de Leva (1996) male mass fractions (see model.MASS_FRACTION).

    torso_length spans hip joint level to the base of the neck. hip_width is the outer
    pelvis width; hip joint centres sit at +/- 0.85 * hip_width / 2. shoulder_width is
    the distance between the two shoulder joints.
    """

    # 기본값 = 키 1.75m, 몸무게 70kg 성인 남성 (단위: m, kg)
    name: str = "baseline"           # 체형 이름 (렌더 파일명 등에 쓰임)
    torso_length: float = 0.645      # 몸통 길이: 고관절 높이 ~ 목 밑
    shoulder_width: float = 0.40     # 양쪽 어깨 관절 사이 거리
    hip_width: float = 0.22          # 골반 바깥 폭 (고관절은 이 값의 85% 지점)
    upper_arm_length: float = 0.325  # 위팔 (어깨 ~ 팔꿈치)
    forearm_length: float = 0.255    # 아래팔 (팔꿈치 ~ 손목)
    thigh_length: float = 0.43       # 허벅지 (고관절 ~ 무릎)
    shin_length: float = 0.43        # 정강이 (무릎 ~ 발목)
    total_mass: float = 70.0     # kg  전체 몸무게. model.py가 부위별 비율로 나눠 줌
    rom_scale: float = 1.0       # joint range of motion multiplier  유연성: 모든 관절 가동범위에 곱함
    strength_scale: float = 1.0  # joint torque ceiling multiplier   근력: 모든 관절 최대 토크에 곱함
    grip_capacity: float = 850.0  # N  악력: 손 하나가 버틸 수 있는 최대 힘 (지치기 전, 손마다)
    # 악력 피로 (contact.py가 매 스텝 적용)
    grip_fatigue_halflife: float = 60.0  # s  악력 한계에 가깝게 매달리면, 최저치 위의 악력이 이 시간마다 절반으로
                                         #    (가볍게 매달리면 하중/악력 비율만큼 느리게 지친다)
    grip_recovery_time: float = 30.0     # s  손을 놓고 쉬면 이 시간이면 거의(95%) 회복
    grip_min_fraction: float = 0.3       #    아무리 지쳐도 처음 악력의 이 비율 밑으로는 안 내려감

    def __post_init__(self) -> None:
        # 잘못된 값(음수 길이, 2m 넘는 팔 등)이 들어오면 바로 에러를 낸다
        for field in ("torso_length", "shoulder_width", "hip_width", "upper_arm_length",
                      "forearm_length", "thigh_length", "shin_length"):
            v = getattr(self, field)
            if not 0.05 <= v <= 2.0:
                raise ValueError(f"{field}={v} is outside [0.05, 2.0] m")
        for field in ("total_mass", "rom_scale", "strength_scale", "grip_capacity",
                      "grip_fatigue_halflife", "grip_recovery_time"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if not 0.0 <= self.grip_min_fraction <= 1.0:
            raise ValueError("grip_min_fraction must be in [0, 1]")

    def with_limbs(self, scale: float, name: str | None = None) -> "BodyProfile":
        """Scale all four limb segments together (e.g. 0.95 = 5% shorter arms and legs).

        팔다리 4마디(위팔·아래팔·허벅지·정강이) 길이에만 scale을 곱한 새 체형을 만든다.
        몸통·머리·폭·몸무게·근력·유연성은 그대로.
        """
        return replace(self, name=name or f"{self.name}_limbs{scale:g}",
                       upper_arm_length=self.upper_arm_length * scale,
                       forearm_length=self.forearm_length * scale,
                       thigh_length=self.thigh_length * scale,
                       shin_length=self.shin_length * scale)

    @property
    def arm_reach(self) -> float:
        return self.upper_arm_length + self.forearm_length

    @property
    def leg_reach(self) -> float:
        return self.thigh_length + self.shin_length


# 미리 만들어 둔 체형 (견본). 실제 실험 설계는 팔만/다리만 따로 바꾸는 것을 권장.
#   baseline    : 기본 175cm / 70kg
#   short_limbs : 팔다리 -5% (키 약 1.71m)
#   long_limbs  : 팔다리 +5% (키 약 1.79m)
BASELINE = BodyProfile()
SHORT_LIMBS = BASELINE.with_limbs(0.95, "short_limbs")
LONG_LIMBS = BASELINE.with_limbs(1.05, "long_limbs")
PRESETS = {p.name: p for p in (BASELINE, SHORT_LIMBS, LONG_LIMBS)}
