"""Bouldering wall and hold layout.

MuJoCo world frame: x = horizontal along the wall, y = wall normal (climber stands at -y),
z = up. The wall's climbing face is the plane y = 0.

[한국어 설명]
볼더링 벽과 홀드 배치(= 볼더링 문제, map)를 정의하는 파일.
좌표: x = 벽을 따라 좌우, y = 벽에서 멀어지는 방향(사람은 -y 쪽에 섬), z = 위. 벽 표면은 y = 0.
새 문제를 만들려면 default_wall()처럼 Wall(holds=(Hold(...), ...))를 하나 더 만들면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Hold:
    """홀드 하나. 지금은 모두 공 모양."""
    name: str              # 이름 (예: "H1"). 그립 이름 grasp_left_H1 등에 쓰임
    x: float               # 벽 위 좌우 위치 (m)
    z: float               # 벽 위 높이 (m)
    radius: float = 0.065  # 공 반지름 (m)
    friction: float = 0.85 # 마찰 계수 (클수록 덜 미끄러짐)


@dataclass(frozen=True)
class Wall:
    """벽 크기와 벽에 붙은 홀드 목록."""
    width: float = 2.3
    height: float = 3.0
    thickness: float = 0.10
    friction: float = 0.95
    holds: tuple[Hold, ...] = field(default_factory=tuple)

    def hold(self, name: str) -> Hold:
        for h in self.holds:
            if h.name == name:
                return h
        raise KeyError(name)


def default_wall() -> Wall:
    """Layout inherited from the first physics prototype so results stay comparable.

    팀원 프로토타입과 같은 홀드 8개 배치 (비교용). 175cm 몸에게는 좁아서 나중에 새 문제로 교체 예정.
    H1, H2 = 발 홀드 / H3~H7 = 손 홀드 / TOP = 꼭대기 목표 홀드
    """
    return Wall(holds=(
        Hold("H1", -0.33, 0.72, 0.075, 0.90),
        Hold("H2", 0.33, 0.72, 0.075, 0.90),
        Hold("H3", -0.38, 1.48, 0.065, 0.82),
        Hold("H4", 0.35, 1.55, 0.060, 0.80),
        Hold("H5", 0.20, 1.62, 0.065, 0.85),
        Hold("H6", -0.20, 0.82, 0.070, 0.90),
        Hold("H7", -0.22, 1.68, 0.065, 0.85),
        Hold("TOP", 0.24, 2.60, 0.075, 0.88),
    ))
