import time
import math
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("arm.xml")
data = mujoco.MjData(model)

#viewer를 열고 움직임은 수동으로 조절
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        start = time.time()

        # 어깨 모터(0번)에 0.2 ~ -0.2 사이 값을 번갈아 준다
        data.ctrl[0] = -0.8 * math.sin(2 * data.time)

        mujoco.mj_step(model, data)   # 시뮬레이션 한 걸음 진행
        viewer.sync()                 # 화면 갱신

        # 실제 시간과 속도를 맞추기
        wait = model.opt.timestep - (time.time() - start)
        if wait > 0:
            time.sleep(wait)