"""TEST-ONLY archived Stage2 normalized PD for NONPHYSICAL software fixtures.

This reproduces historical debug choreography, not Stage3 controller behavior or
physical acquisition/support evidence. Only ctrl is written; native MuJoCo steps
remain responsible for motion, time, and physics results.
"""


def archived_normalized_pose_control(
    model, data, target_pose, kp=None, kd=None, *,
    default_kp=5.0, default_kd=0.5, feedforward=None,
):
    import mujoco

    assert float(model.numeric("contact_mode").data[0]) == 1., (
        "NONPHYSICAL archived controller requires compiled IDEALIZED_DEBUG mode 1"
    )
    assert not feedforward, "Archived normalized PD has no feedforward law"
    kp = default_kp if kp is None else kp
    kd = default_kd if kd is None else kd
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not jname:
            continue
        qadr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        u = kp * (target_pose.get(jname, 0.0) - float(data.qpos[qadr])) - kd * float(data.qvel[vadr])
        ctrl_min, ctrl_max = float(model.actuator_ctrlrange[i, 0]), float(model.actuator_ctrlrange[i, 1])
        data.ctrl[i] = max(ctrl_min, min(ctrl_max, u))
