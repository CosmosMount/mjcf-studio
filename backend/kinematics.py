"""Bounded, transactional closed-chain pose projection for mouse joint dragging."""

import mujoco as mj
import numpy as np
from scipy.optimize import least_squares

# Native equality coordinates (m for connect, dimensionless for weld/joint),
# and tendon lengths (m). This is a kinematic feasibility threshold, not a
# physical contact or dynamic solver tolerance.
PROJECTION_TOLERANCE = 1e-6
MAX_EVALUATIONS = 80
# Keep underdetermined chains near the previous pose. Feasibility is checked
# independently below, so this soft preference cannot admit an open loop.
POSE_REGULARIZATION = 1e-4


def project_joint(model, data, name, requested):
    return _project(model, data, name, requested)


def align_joints(model, data, first, second, frame="world", drivers=None):
    if first == second:
        raise ValueError("请选择两个不同关节")
    pair = [model.joint(n).id for n in (first, second)]
    frame_id = model.body(frame).id
    if not drivers:
        raise ValueError("请选择允许调整的关节")
    ids = [model.joint(n).id for n in drivers]
    if any(model.jnt_type[j] not in (2, 3) for j in ids):
        raise ValueError("配合仅调整 hinge/slide，free/ball 保持固定")
    mj.mj_forward(model, data)
    if np.linalg.norm(data.xanchor[pair[1]] - data.xanchor[pair[0]]) < 1e-9:
        raise ValueError("两个关节中心重合，连线方向未定义")
    return _project(model, data, None, None, (pair, frame_id, ids))


def _project(model, data, name, requested, alignment=None):
    joint = model.joint(name).id if alignment is None else None
    if alignment is None and model.jnt_type[joint] not in (3, 2):
        raise ValueError("鼠标关节拖动仅支持 hinge/slide")
    if alignment is None and not np.isfinite(requested):
        raise ValueError("目标关节读数必须有限")
    if model.nflex:
        raise ValueError("闭链拖动暂不支持含柔性体的模型")
    disabled = int(mj.mjtDisableBit.mjDSBL_CONSTRAINT) | int(
        mj.mjtDisableBit.mjDSBL_EQUALITY
    )
    if model.neq and np.any(data.eq_active) and int(model.opt.disableflags) & disabled:
        raise ValueError("模型已禁用等式约束，无法执行保持闭链的拖动")
    address = model.jnt_qposadr[joint] if alignment is None else None
    target = float(requested) if alignment is None else None
    if alignment is None:
        if model.jnt_limited[joint]:
            target = float(np.clip(target, *model.jnt_range[joint]))
        adjustable = np.flatnonzero(
            np.isin(model.jnt_type, [3, 2]) & (np.arange(model.njnt) != joint)
        )
    else:
        adjustable = np.asarray(sorted(set(alignment[2])), dtype=int)
    addresses = model.jnt_qposadr[adjustable]
    lower = np.where(
        model.jnt_limited[adjustable], model.jnt_range[adjustable, 0], -np.inf
    )
    upper = np.where(
        model.jnt_limited[adjustable], model.jnt_range[adjustable, 1], np.inf
    )
    # A joint locked by equal bounds is held, not passed to scipy's strict bounds.
    locked = lower == upper
    work = mj.MjData(model)
    work.qpos[:] = data.qpos
    if alignment is None:
        work.qpos[address] = target
    work.qpos[addresses[locked]] = lower[locked]
    work.mocap_pos[:] = data.mocap_pos
    work.mocap_quat[:] = data.mocap_quat
    work.eq_active[:] = data.eq_active
    addresses, lower, upper = addresses[~locked], lower[~locked], upper[~locked]
    initial = np.clip(data.qpos[addresses], lower, upper)
    limited_tendons = np.flatnonzero(model.tendon_limited)

    def residual(x):
        work.qpos[addresses] = x
        # Contacts are deliberately absent: dragging is constrained kinematics,
        # not a force perturbation or a promise of collision-free motion.
        mj.mj_kinematics(model, work)
        mj.mj_comPos(model, work)
        mj.mj_tendon(model, work)
        mj.mj_makeConstraint(model, work)
        equality = work.efc_pos[work.efc_type == int(mj.mjtConstraint.mjCNSTR_EQUALITY)]
        length = work.ten_length[limited_tendons]
        limits = model.tendon_range[limited_tendons]
        violations = length - np.clip(length, limits[:, 0], limits[:, 1])
        terms = [equality, violations]
        # Include fixed scalar joints too: an existing invalid pose cannot pass.
        scalar = np.flatnonzero(np.isin(model.jnt_type, [2, 3]) & model.jnt_limited.astype(bool))
        q = work.qpos[model.jnt_qposadr[scalar]]
        terms.append(q - np.clip(q, model.jnt_range[scalar, 0], model.jnt_range[scalar, 1]))
        if alignment is not None:
            pair, frame_id, _ = alignment
            delta = work.xanchor[pair[1]] - work.xanchor[pair[0]]
            normal = work.xmat[frame_id].reshape(3, 3)[:, 2]
            terms.append(np.array([delta @ normal]))
        return np.concatenate(terms)

    def objective(x):
        # Select the nearby branch without trading away the hard acceptance gate.
        return np.concatenate((residual(x), POSE_REGULARIZATION * (x - initial)))

    if len(addresses):
        solution = least_squares(
            objective,
            initial,
            bounds=(lower, upper),
            max_nfev=MAX_EVALUATIONS,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-12,
        )
        values = solution.x
        evaluations = int(solution.nfev)
    else:
        values, evaluations = initial, 1
    error = residual(values)
    maximum = float(np.max(np.abs(error))) if len(error) else 0.0
    accepted = bool(
        np.isfinite(work.qpos).all()
        and np.isfinite(maximum)
        and maximum <= PROJECTION_TOLERANCE
    )
    if alignment is not None:
        pair = alignment[0]
        accepted = bool(accepted and np.linalg.norm(work.xanchor[pair[1]] - work.xanchor[pair[0]]) > 1e-9)
    if accepted:
        # Only commit a feasible configuration. Never leak optimizer candidates.
        data.qpos[:] = work.qpos
        data.qvel[:] = 0
        data.qacc_warmstart[:] = 0
        mj.mj_forward(model, data)
    return {
        "accepted": accepted,
        "joint": name,
        "requested": requested,
        "value": float(data.qpos[address]) if alignment is None else None,
        "limitedTarget": target,
        "residual": maximum,
        "tolerance": PROJECTION_TOLERANCE,
        "evaluations": evaluations,
        "message": ("水平配合已满足；可保存参考姿态或标定指定关节" if alignment is not None else "闭链约束已满足")
        if accepted
        else "目标不可达或求解未收敛；保持上一姿态（请检查初始闭链、限位和肌腱行程）",
    }
