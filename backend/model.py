from __future__ import annotations
import copy
import base64
import warnings
import hashlib
import io
import json
from pathlib import Path
import zipfile
import tempfile
import numpy as np
import mujoco as mj
from lxml import etree
from .geometry import (
    Mesh,
    rotation,
    quaternion,
    valid_inertia,
    combine,
    primitive_properties,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
RUNTIME.mkdir(exist_ok=True)

MATERIALS = [
    {
        "name": "铝（纯铝近似）",
        "density": 2700,
        "rgba": [0.72, 0.76, 0.8, 1],
        "source": "Royal Society of Chemistry / periodic-table/element/13/aluminium",
        "note": "室温近似；合金需按牌号调整",
    },
    {
        "name": "钢（通用估计）",
        "density": 7850,
        "rgba": [0.48, 0.53, 0.58, 1],
        "source": "Engineering Toolbox / metal-alloys-densities-d_50.html",
        "note": "钢种不同密度不同",
    },
    {
        "name": "铜",
        "density": 8960,
        "rgba": [0.72, 0.38, 0.22, 1],
        "source": "Royal Society of Chemistry / periodic-table/element/29/copper",
        "note": "室温纯铜近似",
    },
    {
        "name": "ABS（估计）",
        "density": 1040,
        "rgba": [0.24, 0.45, 0.62, 1],
        "source": "Engineering Toolbox / density-plastics-d_981.html",
        "note": "填料与牌号影响密度",
    },
    {
        "name": "橡胶（估计）",
        "density": 1100,
        "rgba": [0.12, 0.14, 0.16, 1],
        "source": "Engineering Toolbox / density-materials-d_1652.html",
        "note": "配方差异大，建议实测；不修改摩擦",
    },
    {
        "name": "木材（估计）",
        "density": 650,
        "rgba": [0.62, 0.40, 0.23, 1],
        "source": "Engineering Toolbox / wood-density-d_40.html",
        "note": "树种与含水率影响显著",
    },
]

COLLECTIONS = {
    "body": "bodies",
    "geom": "geoms",
    "joint": "joints",
    "site": "sites",
    "camera": "cameras",
    "light": "lights",
    "mesh": "meshes",
    "material": "materials",
    "texture": "textures",
    "equality": "equalities",
    "tendon": "tendons",
    "actuator": "actuators",
    "sensor": "sensors",
    "key": "keys",
}
FIELDS = {
    "body": ["pos", "quat", "mass", "ipos", "iquat", "inertia", "gravcomp"],
    "geom": [
        "pos",
        "quat",
        "size",
        "rgba",
        "material",
        "contype",
        "conaffinity",
        "friction",
        "group",
    ],
    "joint": [
        "pos",
        "axis",
        "range",
        "limited",
        "ref",
        "springref",
        "stiffness",
        "damping",
        "armature",
        "frictionloss",
    ],
    "site": ["pos", "quat", "size", "rgba"],
    "equality": ["active", "name1", "name2", "data", "solref", "solimp"],
    "tendon": [
        "range",
        "limited",
        "stiffness",
        "damping",
        "springlength",
        "frictionloss",
        "width",
        "rgba",
    ],
    "actuator": [
        "target",
        "refsite",
        "gear",
        "ctrlrange",
        "ctrllimited",
        "forcerange",
        "forcelimited",
        "gainprm",
        "biasprm",
        "dynprm",
        "lengthrange",
    ],
    "sensor": ["objname", "refname"],
    "key": ["qpos", "qvel", "ctrl", "act", "time"],
    "material": [
        "rgba",
        "specular",
        "shininess",
        "reflectance",
        "texrepeat",
        "texuniform",
    ],
    "camera": ["pos", "quat", "fovy"],
}


def plain(x):
    if x is None:
        return None
    if hasattr(x, "signature") and hasattr(x, "name"):
        return x.name
    if isinstance(x, (str, bool, int, float)):
        return x
    if isinstance(x, np.ndarray):
        return np.nan_to_num(x).tolist()
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return list(x)
        except TypeError:
            return str(x)


def normalize(s):
    # MuJoCo serialization resolves defaults, alternate orientations, frames and angular units.
    s.compile()
    assets = dict(s.assets)
    normalized = mj.MjSpec.from_string(s.to_xml(), assets=assets)
    for kind, collection in COLLECTIONS.items():
        used = {v.name for v in getattr(normalized, collection)}
        for i, value in enumerate(getattr(normalized, collection)):
            if not value.name:
                name = f"{kind}_{i}"
                while name in used:
                    name += "_"
                value.name = name
                used.add(name)
    return normalized


def load_spec(path: Path, resource_root: Path):
    # Resolve include/resource paths before native parsing can read outside the project.
    visited = set()

    def inspect_xml(filename):
        filename = filename.resolve()
        if not filename.is_relative_to(resource_root.resolve()):
            raise ValueError("XML include 位于工程目录外")
        if filename in visited:
            raise ValueError("XML include 存在循环")
        visited.add(filename)
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        tree = etree.parse(str(filename), parser)
        if tree.docinfo.doctype:
            raise ValueError("不支持 XML DTD/外部实体")
        for node in tree.iter():
            if not isinstance(node.tag, str):
                continue
            for attr in ("file", "meshdir", "texturedir", "assetdir"):
                value = node.get(attr)
                if value and (Path(value).is_absolute() or ".." in Path(value).parts):
                    # Relative traversal is accepted only while remaining inside the bundle.
                    if (
                        not (filename.parent / value)
                        .resolve()
                        .is_relative_to(resource_root.resolve())
                    ):
                        raise ValueError("资源路径位于工程目录外: " + value)
            if node.tag == "include":
                inspect_xml(filename.parent / node.get("file", ""))
        visited.remove(filename)

    inspect_xml(path)
    s = mj.MjSpec.from_file(str(path))
    s.compile()  # never discard unsupported or invalid source elements
    for collection, directory in [
        ("meshes", s.meshdir),
        ("textures", s.texturedir),
        ("hfields", s.meshdir),
    ]:
        for item in getattr(s, collection):
            if not item.file:
                continue
            original = Path(item.file)
            candidates = [
                path.parent / directory / original,
                path.parent / original,
                resource_root / directory / original,
                resource_root / original,
            ]
            source = next((p.resolve() for p in candidates if p.is_file()), None)
            if source is None or not source.is_relative_to(resource_root.resolve()):
                raise ValueError(f"资源缺失或位于工程目录外: {item.file}")
            content = source.read_bytes()
            name = (
                "assets/"
                + hashlib.sha256(content).hexdigest()[:20]
                + source.suffix.lower()
            )
            s.assets[name] = content
            item.file = name
    s.meshdir = ""
    s.texturedir = ""
    s.modelfiledir = ""
    return normalize(s)


class Editor:
    def __init__(self, spec=None):
        self.source_path = None
        self.source_digest = None
        self.spec = normalize(
            spec
            or mj.MjSpec.from_string(
                '<mujoco model="新建模型"><option timestep="0.002"/><worldbody><light name="sun" pos="0 0 4"/><geom name="ground" type="plane" size="3 3 .1" rgba=".17 .20 .24 1"/></worldbody></mujoco>'
            )
        )
        self.meta = {"meshes": {}, "stale": [], "materials": {}, "inertia_sources": {}}
        self.base = self.spec.to_xml()
        self.base_assets = dict(self.spec.assets)
        self.journal = []
        self.revision = 0
        self.undo_stack = []
        self.redo_stack = []
        self.running = False
        self.model = None
        self.data = None
        self.error = None
        self.compile()
        for geom in self.spec.geoms:
            if geom.type == mj.mjtGeom.mjGEOM_MESH:
                self.meta["meshes"][geom.meshname] = self.geom_mesh(geom.name).json()

    def compile(self):
        self.running = False
        try:
            self.model = self.spec.compile()
            self.data = mj.MjData(self.model)
            mj.mj_forward(self.model, self.data)
            self.error = None
        except Exception as exc:
            self.model = self.data = None
            self.error = str(exc)

    def snapshot(self):
        return self.spec.copy(), copy.deepcopy(self.meta), copy.deepcopy(self.journal)

    def restore(self, state):
        self.spec, self.meta, self.journal = state
        self.compile()

    def element(self, target):
        kind, name = target.split(":", 1)
        if kind not in COLLECTIONS:
            raise ValueError("未知元素类型")
        # Native name lookup caches are rebuilt at compilation. Enumerating the spec
        # also finds elements created earlier in the same, not-yet-compiled transaction.
        value = next(
            (v for v in getattr(self.spec, COLLECTIONS[kind]) if v.name == name), None
        )
        if value is None:
            raise ValueError(f"对象不存在: {target}")
        return kind, value

    def command(self, revision, op, target="", args=None):
        if revision != self.revision:
            raise ValueError("模型已更新，请刷新后重试")
        if op in ("undo", "redo"):
            source, destination = (
                (self.undo_stack, self.redo_stack)
                if op == "undo"
                else (self.redo_stack, self.undo_stack)
            )
            if not source:
                raise ValueError("没有可撤销/重做的操作")
            destination.append(self.snapshot())
            self.restore(source.pop())
        else:
            previous = self.snapshot()
            previous_data = self.data
            self.spec = self.spec.copy()
            self.meta = copy.deepcopy(self.meta)
            try:
                self.apply(op, target, args or {})
                self.journal.append({"op": op, "target": target, "args": args or {}})
                self.compile()
                if op == "save_pose" and self.data is not None:
                    self.data.qpos[:] = args["qpos"]
                    mj.mj_forward(self.model, self.data)
                if op == "zero" and previous_data is not None and self.data is not None:
                    self.data.qpos[:] = previous_data.qpos
                    self.data.qvel[:] = previous_data.qvel
                    self.data.ctrl[:] = previous_data.ctrl
                    self.data.act[:] = previous_data.act
                    self.data.mocap_pos[:] = previous_data.mocap_pos
                    self.data.mocap_quat[:] = previous_data.mocap_quat
                    self.data.time = previous_data.time
                    _, joint_name = target.split(":", 1)
                    address = self.model.jnt_qposadr[self.model.joint(joint_name).id]
                    self.data.qpos[address] -= float(args["offset"])
                    mj.mj_forward(self.model, self.data)
                if self.model is not None:
                    for geom in self.spec.geoms:
                        if (
                            geom.type == mj.mjtGeom.mjGEOM_MESH
                            and geom.meshname not in self.meta["meshes"]
                        ):
                            self.meta["meshes"][geom.meshname] = self.geom_mesh(
                                geom.name
                            ).json()
            except Exception:
                self.restore(previous)
                raise
            self.undo_stack.append(previous)
            self.undo_stack = self.undo_stack[-40:]
            self.redo_stack.clear()
        self.revision += 1
        return self.scene()

    def apply(self, op, target, args):
        if op == "set":
            kind, obj = self.element(target)
            if kind == "body" and obj.name == "world":
                raise ValueError("世界坐标系不可移动或设置惯性")
            for key, value in args.items():
                if key not in FIELDS.get(kind, []):
                    raise ValueError(f"不允许编辑字段: {kind}.{key}")
                if (
                    isinstance(value, (list, float, int))
                    and not np.isfinite(value).all()
                ):
                    raise ValueError("参数必须是有限数")
                if key == "axis" and np.linalg.norm(value) < 1e-12:
                    raise ValueError("关节轴不能为零")
                setattr(obj, key, value)
            if kind == "geom" and any(k in args for k in ("pos", "quat", "size")):
                self.stale(obj.parent.name)
        elif op == "add_joint":
            previous = self.spec.compile()
            body = self.spec.body(args["body"])
            if body is None or body.name == "world":
                raise ValueError("关节必须属于非世界刚体")
            if self.spec.joint(args["name"]) is not None:
                raise ValueError("关节名称重复")
            j = body.add_joint(
                name=args["name"],
                type=mj.mjtJoint.mjJNT_HINGE
                if args.get("type", "hinge") == "hinge"
                else mj.mjtJoint.mjJNT_SLIDE,
            )
            j.pos = args.get("pos", [0, 0, 0])
            j.axis = args.get("axis", [0, 0, 1])
            if args.get("space") == "world":
                model = self.spec.compile()
                data = mj.MjData(model)
                mj.mj_forward(model, data)
                bid = model.body(body.name).id
                r = data.xmat[bid].reshape(3, 3)
                j.pos = r.T @ (np.asarray(j.pos) - data.xpos[bid])
                j.axis = r.T @ j.axis
            if np.linalg.norm(j.axis) < 1e-12:
                raise ValueError("关节轴不能为零")
            j.range = args.get("range", [-np.pi, np.pi])
            j.limited = True
            self.remap_keys(previous)
        elif op == "save_pose":
            q = np.asarray(args["qpos"], dtype=float)
            model = self.spec.compile()
            if q.shape != (model.nq,) or not np.isfinite(q).all():
                raise ValueError("参考姿态维度错误或包含非有限数")
            key = self.spec.add_key(name=self.unique("key", args.get("name", "calibrated")))
            key.qpos = q
        elif op == "reframe":
            self.reframe(target, args)
        elif op == "zero":
            self.zero(target, args)
        elif op == "pose_reference":
            _, body = self.element(target)
            body.pos = args["pos"]
            body.quat = args["quat"]
            self.spec.compile()
        elif op == "replace_tendon":
            _, old = self.element(target)
            root = etree.fromstring(self.spec.to_xml().encode())
            element = next(e for e in root.find("tendon") if e.get("name") == old.name)
            path = args["path"]
            types = {w["type"] for w in path}
            if not path or ("joint" in types and types != {"joint"}):
                raise ValueError("固定肌腱只接受 joint，空间肌腱接受 site/geom/pulley")
            element.tag = "fixed" if types == {"joint"} else "spatial"
            for child in list(element):
                element.remove(child)
            for wrap in path:
                typ = wrap["type"]
                attrs = {}
                if typ == "joint":
                    attrs = {"joint": wrap["target"], "coef": str(float(wrap["coef"]))}
                elif typ == "site":
                    attrs = {"site": wrap["target"]}
                elif typ == "geom":
                    attrs = {"geom": wrap["target"]}
                    if wrap.get("sidesite"):
                        attrs["sidesite"] = wrap["sidesite"]
                elif typ == "pulley":
                    attrs = {"divisor": str(float(wrap["divisor"]))}
                else:
                    raise ValueError("未知路径类型")
                etree.SubElement(element, typ, attrs)
            replacement = mj.MjSpec.from_string(
                etree.tostring(root).decode(), assets=dict(self.spec.assets)
            )
            replacement.compile()
            self.spec = replacement
        elif op == "attach_model":
            child = mj.MjSpec.from_string(
                args["xml"],
                assets={k: base64.b64decode(v) for k, v in args["assets"].items()},
            )
            # Native subtree attachment transfers element references, not the Python
            # virtual-file dictionary. The parent owns resources after attachment.
            for filename, content in child.assets.items():
                if (
                    filename in self.spec.assets
                    and self.spec.assets[filename] != content
                ):
                    raise ValueError("导入资源名称冲突: " + filename)
                self.spec.assets[filename] = content
            prefix = args.get("prefix", "imported_")
            if not prefix:
                raise ValueError("导入模型必须指定名称前缀")
            self.spec.copy_during_attach = True
            frame = self.spec.worldbody.add_frame(
                name=self.unique("frame", prefix + "frame")
            )
            self.spec.attach(child, prefix=prefix, frame=frame)
            with warnings.catch_warnings(record=True) as notices:
                warnings.simplefilter("always")
                self.spec = normalize(self.spec)
            self.meta.setdefault("notes", []).extend(
                sorted(set(str(n.message) for n in notices))
            )
            metadata = args.get("meta", {})
            for name in metadata.get("stale", []):
                self.stale(prefix + name)
            for category in ("materials", "inertia_sources"):
                for name, value in metadata.get(category, {}).items():
                    self.meta[category][prefix + name] = copy.deepcopy(value)
            for category in ("generated_meshes", "generated_materials"):
                self.meta.setdefault(category, []).extend(
                    prefix + n for n in metadata.get(category, [])
                )
        elif op == "add_obj":
            self.add_obj(args)
        elif op == "split":
            if args.get("preview_revision", self.revision) != self.revision:
                raise ValueError("预览已过期，请重新生成")
            self.split(target, args)
        elif op == "cap":
            if args.get("preview_revision", self.revision) != self.revision:
                raise ValueError("预览已过期，请重新生成")
            _, geom = self.element(target)
            result = self.geom_mesh(geom.name).cap()
            self.install_mesh(geom, result, geom.meshname + "_cap")
            self.stale(geom.parent.name)
        elif op == "inertia":
            self.inertia(target, args)
        elif op == "cap_material":
            self.cap_material(target, args)
        elif op == "material":
            self.material(target, args)
        elif op == "batch":
            for command in args["commands"]:
                self.apply(
                    command["op"], command.get("target", ""), command.get("args", {})
                )
            self.spec.compile()
        else:
            raise ValueError(f"未知操作: {op}")

    def stale(self, body):
        if body not in self.meta["stale"]:
            self.meta["stale"].append(body)

    def reframe(self, target, args):
        kind, body = self.element(target)
        if kind != "body" or body.name == "world":
            raise ValueError("请选择非世界刚体")
        model = self.spec.compile()
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        bid = model.body(body.name).id
        old_r = rotation(body.quat)
        new_pos, new_r = np.asarray(args["pos"], float), rotation(args["quat"])
        if args.get("space") == "world":
            parent = model.body_parentid[bid]
            pr = data.xmat[parent].reshape(3, 3)
            new_pos = pr.T @ (new_pos - data.xpos[parent])
            new_r = pr.T @ new_r
        # x_new = Rnew^T (Rold x_old + pold - pnew)
        r = new_r.T @ old_r
        t = new_r.T @ (np.asarray(body.pos) - new_pos)
        delta = -r.T @ t
        # Quaternion coordinates are basis-dependent. Scalar joints carry their basis
        # in axis; ball/free joints instead require a coordinate change in keyframes.
        for joint in body.joints:
            jid = model.joint(joint.name).id
            qa, va = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
            if joint.type not in (mj.mjtJoint.mjJNT_FREE, mj.mjtJoint.mjJNT_BALL):
                continue
            for actuator in self.spec.actuators:
                if actuator.target != joint.name:
                    continue
                gear = actuator.gear.copy()
                if joint.type == mj.mjtJoint.mjJNT_FREE:
                    if (
                        np.linalg.norm(delta) > 1e-12
                        and np.linalg.norm(gear[:3]) > 1e-12
                    ):
                        raise ValueError(
                            "移动自由关节原点不能用常量 gear 等价表示此执行器的力矩，请先调整传动: actuator:"
                            + actuator.name
                        )
                    gear[3:6] = r @ gear[3:6]
                else:
                    gear[:3] = r @ gear[:3]
                actuator.gear = gear
            for key in self.spec.keys:
                q = np.asarray(key.qpos).copy()
                v = np.asarray(key.qvel).copy()
                if not len(q):
                    q = model.qpos0.copy()
                if joint.type == mj.mjtJoint.mjJNT_FREE:
                    qr = rotation(q[qa + 3 : qa + 7])
                    q[qa : qa + 3] += qr @ delta
                    q[qa + 3 : qa + 7] = quaternion(qr @ r.T)
                    if len(v):
                        v[va : va + 3] += qr @ np.cross(v[va + 3 : va + 6], delta)
                        v[va + 3 : va + 6] = r @ v[va + 3 : va + 6]
                else:
                    q[qa : qa + 4] = quaternion(r @ rotation(q[qa : qa + 4]) @ r.T)
                    if len(v):
                        v[va : va + 3] = r @ v[va : va + 3]
                key.qpos = q
                if len(v):
                    key.qvel = v
        if body.mocap:
            mid = int(model.body_mocapid[bid])
            for key in self.spec.keys:
                positions = np.asarray(key.mpos).copy().reshape(-1, 3)
                quaternions = np.asarray(key.mquat).copy().reshape(-1, 4)
                if len(positions) and len(quaternions):
                    qr = rotation(quaternions[mid])
                    positions[mid] += qr @ delta
                    quaternions[mid] = quaternion(qr @ r.T)
                    key.mpos = positions.ravel()
                    key.mquat = quaternions.ravel()
        for child in [*body.bodies, *body.geoms, *body.sites, *body.cameras]:
            child.pos = r @ child.pos + t
            child.quat = quaternion(r @ rotation(child.quat))
        for light in body.lights:
            light.pos = r @ light.pos + t
            light.dir = r @ light.dir
        for joint in body.joints:
            joint.pos = r @ joint.pos + t
            joint.axis = r @ joint.axis
        # Pin inferred inertia before changing the frame.
        body.mass = model.body_mass[bid]
        body.ipos = r @ model.body_ipos[bid] + t
        body.iquat = quaternion(r @ rotation(model.body_iquat[bid]))
        body.inertia = model.body_inertia[bid]
        body.explicitinertial = True
        for eq in self.spec.equalities:
            if eq.objtype != mj.mjtObj.mjOBJ_BODY:
                continue
            d = eq.data.copy()
            if eq.type == mj.mjtEq.mjEQ_CONNECT:
                # User anchor lives in body1; body2 anchor is compiled from reference pose.
                if eq.name1 == body.name:
                    d[:3] = r @ d[:3] + t
            elif eq.type == mj.mjtEq.mjEQ_WELD:
                # Explicit relpose needs a basis change on the corresponding side.
                if np.linalg.norm(d[6:10]) > 0:
                    er = rotation(d[6:10])
                    ep = d[3:6].copy()
                    if eq.name1 == body.name:
                        ep = r @ ep + t
                        er = r @ er
                    if eq.name2 == body.name:
                        ep = ep - er @ r.T @ t
                        er = er @ r.T
                    d[3:6] = ep
                    d[6:10] = quaternion(er)
                if eq.name2 == body.name:
                    d[:3] = r @ d[:3] + t
            eq.data = d
        body.pos = new_pos
        body.quat = quaternion(new_r)
        self.spec.compile()

    def zero(self, target, args):
        kind, joint = self.element(target)
        if kind != "joint" or joint.type not in (
            mj.mjtJoint.mjJNT_HINGE,
            mj.mjtJoint.mjJNT_SLIDE,
        ):
            raise ValueError("零点标定仅适用于 hinge/slide")
        offset = float(args["offset"])
        if not np.isfinite(offset):
            raise ValueError("偏移必须有限")
        model = self.spec.compile()
        jid = model.joint(joint.name).id
        qadr = model.jnt_qposadr[jid]
        blockers = []
        for tendon in self.spec.tendons:
            if any(
                w.type == mj.mjtWrap.mjWRAP_JOINT
                and getattr(w.target, "name", w.target) == joint.name
                for w in tendon.path
            ):
                blockers.append("tendon:" + tendon.name)
        for actuator in self.spec.actuators:
            if actuator.target == joint.name and (
                actuator.biastype != mj.mjtBias.mjBIAS_NONE
                or actuator.dyntype != mj.mjtDyn.mjDYN_NONE
            ):
                blockers.append("actuator:" + actuator.name)
        resolved = args.get("resolved", {})
        missing = [b for b in blockers if b not in resolved]
        if missing:
            raise ValueError(
                "不能自动保证等价；请同事务提供关联参数更新: " + ", ".join(missing)
            )
        joint.ref -= offset
        joint.springref -= offset
        joint.range = np.asarray(joint.range) - offset
        for key in self.spec.keys:
            qpos = np.asarray(key.qpos).copy()
            if len(qpos):
                qpos[qadr] -= offset
                key.qpos = qpos
        # equality/joint uses q-ref, which remains invariant; polynomial unchanged.
        for related, fields in resolved.items():
            self.apply("set", related, fields)
        self.spec.compile()

    def unique(self, kind, stem):
        stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem) or kind
        name = stem
        n = 1
        used = {v.name for v in getattr(self.spec, COLLECTIONS.get(kind, kind + "s"))}
        while name in used:
            name = f"{stem}_{n}"
            n += 1
        return name

    def install_mesh(self, geom, mesh, stem):
        old_name = geom.meshname
        name = self.unique("mesh", stem)
        asset = self.spec.add_mesh(name=name)
        # Native compilation needs a well-defined centering inertia even for an open
        # material patch. Keep its authored triangles; this is only native preview
        # inference. The body's stale gate forbids simulation/export until explicit
        # physical inertia is computed from closed source surfaces or supplied manually.
        asset.inertia = mj.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        # OBJ carries independent per-corner UV/normal indices; inline MJCF cannot round-trip them.
        vertices = mesh.vertices
        if len(vertices) == 3:
            vertices = np.vstack([vertices, vertices.mean(axis=0)])
        if np.linalg.matrix_rank(vertices - vertices.mean(axis=0)) < 3:
            asset.inertia = mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
            geom.mass = 0
        lines = ["v " + " ".join(format(float(x), ".17g") for x in v) for v in vertices]
        if mesh.uv is not None:
            lines += [
                "vt " + " ".join(format(float(x), ".17g") for x in uv)
                for uv in mesh.uv.reshape(-1, 2)
            ]
        if mesh.normals is not None:
            lines += [
                "vn " + " ".join(format(float(x), ".17g") for x in n)
                for n in mesh.normals.reshape(-1, 3)
            ]
        for i, face in enumerate(mesh.faces):
            corners = []
            for j, v in enumerate(face):
                corner = str(int(v) + 1)
                if mesh.uv is not None or mesh.normals is not None:
                    corner += "/" + (str(i * 3 + j + 1) if mesh.uv is not None else "")
                if mesh.normals is not None:
                    corner += "/" + str(i * 3 + j + 1)
                corners.append(corner)
            lines.append("f " + " ".join(corners))
        raw = ("\n".join(lines) + "\n").encode()
        filename = "assets/" + hashlib.sha256(raw).hexdigest()[:20] + ".obj"
        self.spec.assets[filename] = raw
        asset.file = filename
        geom.type = mj.mjtGeom.mjGEOM_MESH
        geom.meshname = name
        self.meta["meshes"][name] = mesh.json()
        generated = self.meta.setdefault("generated_meshes", [])
        generated.append(name)
        if old_name in generated and not any(
            g.meshname == old_name for g in self.spec.geoms
        ):
            old = next((a for a in self.spec.meshes if a.name == old_name), None)
            if old is not None:
                old_file = old.file
                self.spec.delete(old)
                self.drop_unused_asset(old_file)
            generated.remove(old_name)
            self.meta["meshes"].pop(old_name, None)

    def drop_unused_asset(self, filename):
        if not filename:
            return
        for collection in ("meshes", "textures", "hfields", "skins"):
            for item in getattr(self.spec, collection):
                if getattr(item, "file", "") == filename or filename in getattr(
                    item, "cubefiles", []
                ):
                    return
        self.spec.assets.pop(filename, None)

    def add_obj(self, args):
        mesh = Mesh.from_json(args["mesh"])
        scale = float(args.get("scale", 1))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("OBJ 尺度必须为正")
        mesh.vertices *= scale
        name = args.get("name", "part")
        body = self.spec.body(args["body"]) if args.get("body") else None
        if body is None:
            name = self.unique("body", name)
            body = self.spec.worldbody.add_body(
                name=name, pos=args.get("pos", [0, 0, 0])
            )
        geom = body.add_geom(
            name=self.unique("geom", name + "_visual"), contype=0, conaffinity=0
        )
        self.install_mesh(geom, mesh, name + "_mesh")
        self.stale(body.name)

    def geom_mesh(self, name):
        geom = self.spec.geom(name)
        if geom is None or geom.type != mj.mjtGeom.mjGEOM_MESH:
            raise ValueError("请选择 mesh 几何体")
        if geom.meshname in self.meta["meshes"]:
            return Mesh.from_json(self.meta["meshes"][geom.meshname])
        model = self.spec.compile()
        mid = model.mesh(geom.meshname).id
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        # Reconstruct geom-local coordinates (compiled arrays are inertia-centered).
        vertices = (
            model.mesh_vert[va : va + vn] @ rotation(model.mesh_quat[mid]).T
            + model.mesh_pos[mid]
        )
        faces = model.mesh_face[fa : fa + fn].copy()
        uv = None
        if model.mesh_texcoordnum[mid] > 0:
            ta = model.mesh_texcoordadr[mid]
            ids = model.mesh_facetexcoord[fa : fa + fn]
            if np.all(ids >= 0):
                uv = model.mesh_texcoord[ta + ids]
        na = model.mesh_normaladr[mid]
        ni = model.mesh_facenormal[fa : fa + fn]
        normals = model.mesh_normal[na + ni] @ rotation(model.mesh_quat[mid]).T
        return Mesh(vertices, faces, uv, normals)

    def remap_keys(self, previous):
        saved = [
            (key, np.asarray(key.qpos).copy(), np.asarray(key.qvel).copy())
            for key in self.spec.keys
        ]
        if not saved:
            return
        for key, _, _ in saved:
            key.qpos = []
            key.qvel = []
        current = self.spec.compile()
        for key, old_q, old_v in saved:
            q = current.qpos0.copy()
            v = np.zeros(current.nv)
            for i in range(previous.njnt):
                name = previous.joint(i).name
                try:
                    j = current.joint(name).id
                except KeyError:
                    continue
                nq, nv = {0: (7, 6), 1: (4, 3), 2: (1, 1), 3: (1, 1)}[
                    int(previous.jnt_type[i])
                ]
                a, b = int(previous.jnt_qposadr[i]), int(current.jnt_qposadr[j])
                c, d = int(previous.jnt_dofadr[i]), int(current.jnt_dofadr[j])
                if len(old_q):
                    q[b : b + nq] = old_q[a : a + nq]
                if len(old_v):
                    v[d : d + nv] = old_v[c : c + nv]
            key.qpos = q
            key.qvel = v

    def split(self, target, args):
        _, original = self.element(target)
        first, remaining = self.geom_mesh(original.name).split(args["faces"])
        if args.get("cap"):
            first = first.cap()
            remaining = remaining.cap()
        parent_name = original.parent.name
        old_mesh = original.meshname
        previous = self.spec.compile()
        root = etree.fromstring(self.spec.to_xml().encode())
        source = root.xpath(".//geom[@name=$name]", name=original.name)[0]
        parent = source.getparent()
        destination = parent
        if args.get("new_body", True):
            child_name = self.unique("body", args.get("name", "segment"))
            destination = etree.SubElement(parent, "body", name=child_name)
            axis = np.asarray(args.get("axis", [0, 0, 1]), float)
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError("关节轴不能为零")
            etree.SubElement(
                destination,
                "joint",
                name=self.unique("joint", child_name + "_hinge"),
                type="hinge",
                axis=" ".join(map(str, axis)),
                pos=" ".join(map(str, args.get("anchor", [0, 0, 0]))),
            )
            for identifier in args.get("move", []):
                kind, obj = self.element(identifier)
                if (
                    kind not in ("body", "site", "geom")
                    or obj.parent.name != parent_name
                    or obj.name == original.name
                ):
                    raise ValueError(
                        "只能显式转移原刚体的直接子结构、site 或其他几何体"
                    )
                node = root.xpath(".//" + kind + "[@name=$name]", name=obj.name)[0]
                node.getparent().remove(node)
                destination.append(node)
        else:
            child_name = parent_name
        clone = copy.deepcopy(source)
        clone_name = self.unique("geom", original.name + "_split")
        clone.set("name", clone_name)
        destination.append(clone)
        # Moving XML nodes preserves contact/sensor/actuator references; deleting Mjs elements
        # would cascade into those dependencies. MjSpec remains the committed model.
        self.spec = mj.MjSpec.from_string(
            etree.tostring(root).decode(), assets=dict(self.spec.assets)
        )
        self.remap_keys(previous)
        self.install_mesh(self.spec.geom(clone_name), first, old_mesh + "_selected")
        self.install_mesh(self.spec.geom(original.name), remaining, old_mesh + "_rest")
        self.stale(parent_name)
        self.stale(child_name)

    def cap_material(self, target, args):
        _, geom = self.element(target)
        mesh = self.geom_mesh(geom.name)
        if mesh.materials is None or len(np.unique(mesh.materials)) < 2:
            raise ValueError("该网格没有可单独着色的补面")
        ids = np.flatnonzero(mesh.materials == mesh.materials.max())
        cap, rest = mesh.split(ids)
        model = self.spec.compile()
        bid = model.body(geom.parent.name).id
        body = geom.parent
        body.mass = model.body_mass[bid]
        body.ipos = model.body_ipos[bid]
        body.inertia = model.body_inertia[bid]
        body.iquat = model.body_iquat[bid]
        body.explicitinertial = True
        new = body.add_geom(
            name=self.unique("geom", geom.name + "_cap"),
            type=mj.mjtGeom.mjGEOM_MESH,
            pos=geom.pos,
            quat=geom.quat,
            contype=0,
            conaffinity=0,
            mass=0,
            material=geom.material,
        )
        self.install_mesh(new, cap, geom.meshname + "_cap_surface")
        self.install_mesh(geom, rest, geom.meshname + "_outer_surface")
        # Surface appearance does not assign a second material volume or duplicate collision.
        options = dict(args)
        options["density"] = (
            self.meta["materials"].get(geom.name, {}).get("density", 1000)
        )
        self.material("geom:" + new.name, options)

    def inertia(self, target, args):
        kind, body = self.element(target)
        if kind != "body":
            raise ValueError("惯性属于刚体")
        if body.name == "world":
            raise ValueError("世界刚体没有可编辑的质量惯性")
        if args.get("mode") == "manual":
            result = {k: args[k] for k in ("mass", "center", "tensor")}
            tensor = valid_inertia(
                **{
                    "mass": result["mass"],
                    "center": result["center"],
                    "tensor": result["tensor"],
                }
            )
        else:
            properties = []
            entries = []
            if len(set(args["geoms"])) != len(args["geoms"]):
                raise ValueError("实体不能重复选择")
            for name in args["geoms"]:
                geom = self.spec.geom(name)
                if geom is None or geom.parent.name != body.name:
                    raise ValueError("实体必须属于选中刚体")
                density = float(
                    args.get("densities", {}).get(
                        name,
                        self.meta["materials"].get(name, {}).get("density", 1000)
                        if args.get("mode") == "materials"
                        else args.get("density", 1000),
                    )
                )
                mesh = (
                    self.geom_mesh(name)
                    if geom.type == mj.mjtGeom.mjGEOM_MESH
                    else None
                )
                entries.append((geom, density, mesh))
            open_surfaces = any(
                mesh is not None and not mesh.diagnostics()["valid_solid"]
                for _, _, mesh in entries
            )
            if open_surfaces:
                if (
                    any(mesh is None for _, _, mesh in entries)
                    or len({density for _, density, _ in entries}) != 1
                ):
                    raise ValueError(
                        "开口表面须选择全部组成面，并使用同一实体密度；或使用手动惯性"
                    )
                vertices = []
                faces = []
                offset = 0
                for geom, density, mesh in entries:
                    vertices.append(mesh.vertices @ rotation(geom.quat).T + geom.pos)
                    faces.append(mesh.faces + offset)
                    offset += len(mesh.vertices)
                united = Mesh(np.vstack(vertices), np.vstack(faces))
                properties = [united.mass_properties(entries[0][1])]
            else:
                for geom, density, mesh in entries:
                    p = (
                        mesh.mass_properties(density)
                        if mesh is not None
                        else primitive_properties(int(geom.type), geom.size, density)
                    )
                    r = rotation(geom.quat)
                    p["center"] = (r @ p["center"] + geom.pos).tolist()
                    p["tensor"] = (r @ p["tensor"] @ r.T).tolist()
                    properties.append(p)
            result = combine(properties)
            if args.get("mode") == "measured":
                mass = float(args["mass"])
                if mass <= 0:
                    raise ValueError("实测质量必须大于零")
                result["tensor"] = (
                    np.asarray(result["tensor"]) * mass / result["mass"]
                ).tolist()
                result["mass"] = mass
            tensor = valid_inertia(result["mass"], result["center"], result["tensor"])
        values, vectors = np.linalg.eigh(tensor)
        if np.linalg.det(vectors) < 0:
            vectors[:, 0] *= -1
        body.mass = result["mass"]
        body.ipos = result["center"]
        body.inertia = values
        body.iquat = quaternion(vectors)
        body.fullinertia = [np.nan, 0, 0, 0, 0, 0]
        body.explicitinertial = True
        self.meta["stale"] = [x for x in self.meta["stale"] if x != body.name]
        self.meta["inertia_sources"][body.name] = copy.deepcopy(args)

    def material(self, target, args):
        _, geom = self.element(target)
        previous = geom.material
        name = self.unique("material", geom.name + "_material")
        material = self.spec.add_material(name=name)
        material.rgba = args.get("rgba", [0.7, 0.75, 0.8, 1])
        material.specular = float(args.get("specular", 0.3))
        material.shininess = float(args.get("shininess", 0.3))
        material.reflectance = float(args.get("reflectance", 0))
        material.texrepeat = args.get("repeat", [1, 1])
        if args.get("texture") == "keep":
            old_material = next(
                (m for m in self.spec.materials if m.name == previous), None
            )
            if old_material is not None:
                material.textures = list(old_material.textures)
        elif args.get("texture"):
            texture = self.spec.add_texture(
                name=self.unique("texture", name + "_texture")
            )
            texture.type = mj.mjtTexture.mjTEXTURE_2D
            if args["texture"] == "checker":
                texture.builtin = mj.mjtBuiltin.mjBUILTIN_CHECKER
                texture.width = 256
                texture.height = 256
                texture.rgb1 = [0.8, 0.8, 0.8]
                texture.rgb2 = [0.3, 0.3, 0.3]
            else:
                from PIL import Image

                raw = base64.b64decode(args["texture"])
                image = Image.open(io.BytesIO(raw))
                image.verify()
                image = Image.open(io.BytesIO(raw)).convert("RGB")
                output = io.BytesIO()
                image.save(output, format="PNG")
                encoded = output.getvalue()
                filename = "assets/" + hashlib.sha256(encoded).hexdigest()[:20] + ".png"
                self.spec.assets[filename] = encoded
                texture.file = filename
            material.textures = [""] * 10
            material.textures[1] = texture.name
        geom.material = name
        geom.rgba = [0.5, 0.5, 0.5, 1]  # MuJoCo default delegates to material
        self.meta["materials"][geom.name] = {
            "density": args.get("density", 1000),
            "name": args.get("name", "自定义"),
            "source": args.get("source", "用户输入"),
        }
        generated = self.meta.setdefault("generated_materials", [])
        generated.append(name)
        used = (
            {g.material for g in self.spec.geoms}
            | {t.material for t in self.spec.tendons}
            | {m.material for m in self.spec.meshes}
        )
        if previous in generated and previous not in used:
            old = next(m for m in self.spec.materials if m.name == previous)
            textures = list(old.textures)
            self.spec.delete(old)
            generated.remove(previous)
            used_textures = {t for m in self.spec.materials for t in m.textures}
            for texname in textures:
                if not texname or texname in used_textures:
                    continue
                texture = next(
                    (t for t in self.spec.textures if t.name == texname), None
                )
                if texture is not None:
                    filename = texture.file
                    self.spec.delete(texture)
                    self.drop_unused_asset(filename)

    def relations(self):
        result = []
        for kind in ("equality", "tendon", "actuator", "sensor", "key"):
            for element in getattr(self.spec, COLLECTIONS[kind]):
                item = {
                    "id": kind + ":" + element.name,
                    "name": element.name,
                    "kind": kind,
                    "fields": {
                        k: plain(getattr(element, k))
                        for k in FIELDS.get(kind, [])
                        if hasattr(element, k)
                    },
                }
                if kind == "tendon":
                    item["path"] = [
                        {
                            k: plain(getattr(w, k))
                            for k in ("type", "target", "coef", "divisor", "sidesite")
                        }
                        for w in element.path
                    ]
                if hasattr(element, "type"):
                    item["type"] = str(element.type)
                refs = []
                if kind == "equality":
                    object_kind = {
                        int(mj.mjtObj.mjOBJ_BODY): "body",
                        int(mj.mjtObj.mjOBJ_SITE): "site",
                        int(mj.mjtObj.mjOBJ_JOINT): "joint",
                        int(mj.mjtObj.mjOBJ_TENDON): "tendon",
                    }.get(int(element.objtype))
                    if element.type == mj.mjtEq.mjEQ_JOINT:
                        object_kind = "joint"
                    elif element.type == mj.mjtEq.mjEQ_TENDON:
                        object_kind = "tendon"
                    if object_kind:
                        refs = [
                            object_kind + ":" + n
                            for n in (element.name1, element.name2)
                            if n
                        ]
                elif kind == "tendon":
                    for wrap in element.path:
                        if wrap.type == mj.mjtWrap.mjWRAP_PULLEY:
                            continue
                        k = (
                            "joint"
                            if wrap.type == mj.mjtWrap.mjWRAP_JOINT
                            else "site"
                            if wrap.type == mj.mjtWrap.mjWRAP_SITE
                            else "geom"
                        )
                        refs.append(k + ":" + plain(wrap.target))
                elif kind == "actuator":
                    k = {
                        0: "joint",
                        1: "joint",
                        2: "site",
                        3: "tendon",
                        4: "site",
                        5: "body",
                    }.get(int(element.trntype))
                    if k:
                        refs = [k + ":" + element.target]
                elif kind == "sensor":
                    k = {
                        int(mj.mjtObj.mjOBJ_BODY): "body",
                        int(mj.mjtObj.mjOBJ_SITE): "site",
                        int(mj.mjtObj.mjOBJ_JOINT): "joint",
                        int(mj.mjtObj.mjOBJ_TENDON): "tendon",
                        int(mj.mjtObj.mjOBJ_ACTUATOR): "actuator",
                        int(mj.mjtObj.mjOBJ_GEOM): "geom",
                    }.get(int(element.objtype))
                    if k and element.objname:
                        refs = [k + ":" + element.objname]
                item["references"] = refs
                result.append(item)
        return result

    def scene(self):
        result = {
            "revision": self.revision,
            "sourcePath": self.source_path,
            "name": self.spec.modelname,
            "diagnostics": [],
            "bodies": [],
            "geoms": [],
            "joints": [],
            "sites": [],
            "relations": self.relations(),
            "meta": self.meta,
            "canUndo": bool(self.undo_stack),
            "canRedo": bool(self.redo_stack),
            "materials": MATERIALS,
        }
        if self.error:
            result["diagnostics"].append({"severity": "error", "message": self.error})
        for message in self.meta.get("notes", []):
            result["diagnostics"].append({"severity": "warning", "message": message})
        for name in self.meta["stale"]:
            result["diagnostics"].append(
                {
                    "severity": "warning",
                    "target": "body:" + name,
                    "message": name + " 的几何已改变，请重新确认惯性",
                }
            )
        model, data = self.model, self.data
        for body in self.spec.bodies:
            item = {
                "id": "body:" + body.name,
                "name": body.name,
                "parent": body.parent.name if body.parent else None,
                "fields": {k: plain(getattr(body, k)) for k in FIELDS["body"]},
                "joints": [
                    j.name for j in self.spec.joints if j.parent.name == body.name
                ],
            }
            if model is not None:
                bid = model.body(body.name).id
                item.update(
                    worldPos=data.xpos[bid].tolist(),
                    worldQuat=data.xquat[bid].tolist(),
                    mass=float(model.body_mass[bid]),
                    com=data.xipos[bid].tolist(),
                    inertia=model.body_inertia[bid].tolist(),
                    inertialQuat=quaternion(data.ximat[bid].reshape(3, 3)).tolist(),
                    inertialCenter=model.body_ipos[bid].tolist(),
                    inertialTensor=(
                        rotation(model.body_iquat[bid])
                        @ np.diag(model.body_inertia[bid])
                        @ rotation(model.body_iquat[bid]).T
                    ).tolist(),
                )
            result["bodies"].append(item)
        for geom in self.spec.geoms:
            item = {
                "id": "geom:" + geom.name,
                "name": geom.name,
                "parent": geom.parent.name,
                "type": int(geom.type),
                "fields": {k: plain(getattr(geom, k)) for k in FIELDS["geom"]},
                "mesh": geom.meshname,
            }
            if model is not None:
                gid = model.geom(geom.name).id
                bid = model.geom_bodyid[gid]
                br = data.xmat[bid].reshape(3, 3)
                item["editPos"] = (data.xpos[bid] + br @ geom.pos).tolist()
                item["editQuat"] = quaternion(br @ rotation(geom.quat)).tolist()
                item.update(
                    worldPos=data.geom_xpos[gid].tolist(),
                    worldQuat=quaternion(data.geom_xmat[gid].reshape(3, 3)).tolist(),
                    size=model.geom_size[gid].tolist(),
                    rgba=model.geom_rgba[gid].tolist(),
                )
                mid = model.geom_matid[gid]
                if mid >= 0:
                    mat = self.spec.material(model.material(mid).name)
                    item["appearance"] = {
                        k: plain(getattr(mat, k)) for k in FIELDS["material"]
                    }
                    texture_id = int(model.mat_texid[mid, 1])
                    if texture_id >= 0:
                        item["texture"] = model.texture(texture_id).name
                    if np.allclose(geom.rgba, [0.5, 0.5, 0.5, 1]):
                        item["rgba"] = model.mat_rgba[mid].tolist()
                if geom.type == mj.mjtGeom.mjGEOM_MESH:
                    mi = model.geom_dataid[gid]
                    va = model.mesh_vertadr[mi]
                    vn = model.mesh_vertnum[mi]
                    fa = model.mesh_faceadr[mi]
                    fn = model.mesh_facenum[mi]
                    mesh = {
                        "vertices": model.mesh_vert[va : va + vn].tolist(),
                        "faces": model.mesh_face[fa : fa + fn].tolist(),
                        "uv": None,
                    }
                    if model.mesh_texcoordnum[mi] > 0:
                        ids = model.mesh_facetexcoord[fa : fa + fn]
                        ta = model.mesh_texcoordadr[mi]
                        if np.all(ids >= 0):
                            mesh["uv"] = model.mesh_texcoord[ta + ids].tolist()
                    item["geometry"] = mesh
            result["geoms"].append(item)
        for joint in self.spec.joints:
            item = {
                "id": "joint:" + joint.name,
                "name": joint.name,
                "parent": joint.parent.name,
                "type": int(joint.type),
                "fields": {k: plain(getattr(joint, k)) for k in FIELDS["joint"]},
            }
            if model is not None:
                jid = model.joint(joint.name).id
                item.update(
                    worldPos=data.xanchor[jid].tolist(),
                    worldAxis=data.xaxis[jid].tolist(),
                    qpos=float(data.qpos[model.jnt_qposadr[jid]]),
                    qposAddress=int(model.jnt_qposadr[jid]),
                    effectiveLimited=bool(model.jnt_limited[jid]),
                )
            result["joints"].append(item)
        for site in self.spec.sites:
            item = {
                "id": "site:" + site.name,
                "name": site.name,
                "parent": site.parent.name,
                "fields": {k: plain(getattr(site, k)) for k in FIELDS["site"]},
            }
            if model is not None:
                sid = model.site(site.name).id
                item.update(
                    worldPos=data.site_xpos[sid].tolist(),
                    worldQuat=quaternion(data.site_xmat[sid].reshape(3, 3)).tolist(),
                )
            result["sites"].append(item)
        if model is None:
            # Drafts can contain planar pieces that MuJoCo cannot compile. Render the authored
            # reference transforms so they remain editable; do not present them as simulated.
            transforms = {}
            for item in result["bodies"]:
                p = np.asarray(item["fields"]["pos"])
                r = rotation(item["fields"]["quat"])
                if item["parent"] in transforms:
                    pp, pr = transforms[item["parent"]]
                    p = pp + pr @ p
                    r = pr @ r
                transforms[item["name"]] = (p, r)
                item.update(worldPos=p.tolist(), worldQuat=quaternion(r).tolist())
            for item in [*result["geoms"], *result["sites"], *result["joints"]]:
                p, r = transforms[item["parent"]]
                item["worldPos"] = (p + r @ item["fields"]["pos"]).tolist()
                if item["id"].startswith("joint:"):
                    item["worldAxis"] = (r @ item["fields"]["axis"]).tolist()
                else:
                    item["worldQuat"] = quaternion(
                        r @ rotation(item["fields"]["quat"])
                    ).tolist()
                if item["id"].startswith("geom:"):
                    item["rgba"] = item["fields"]["rgba"]
                    item["size"] = item["fields"]["size"]
                    if item["mesh"] in self.meta["meshes"]:
                        item["geometry"] = self.meta["meshes"][item["mesh"]]
        result["state"] = self.state()
        return result

    def state(self):
        if self.model is None:
            return {"running": False, "valid": False}
        m, d = self.model, self.data
        residual = d.efc_pos[: d.nefc][
            d.efc_type[: d.nefc] == int(mj.mjtConstraint.mjCNSTR_EQUALITY)
        ]
        return {
            "valid": True,
            "running": self.running,
            "time": float(d.time),
            "qpos": d.qpos.tolist(),
            "ctrl": d.ctrl.tolist(),
            "constraintResidual": float(np.max(np.abs(residual)))
            if len(residual)
            else 0.0,
            "geoms": [
                {
                    "name": m.geom(i).name,
                    "pos": d.geom_xpos[i].tolist(),
                    "quat": quaternion(d.geom_xmat[i].reshape(3, 3)).tolist(),
                    "editPos": (
                        d.xpos[m.geom_bodyid[i]]
                        + d.xmat[m.geom_bodyid[i]].reshape(3, 3)
                        @ self.spec.geom(m.geom(i).name).pos
                    ).tolist(),
                    "editQuat": quaternion(
                        d.xmat[m.geom_bodyid[i]].reshape(3, 3)
                        @ rotation(self.spec.geom(m.geom(i).name).quat)
                    ).tolist(),
                }
                for i in range(m.ngeom)
            ],
            "contacts": [d.contact[i].pos.tolist() for i in range(d.ncon)],
            "bodies": [
                {
                    "name": m.body(i).name,
                    "pos": d.xpos[i].tolist(),
                    "quat": d.xquat[i].tolist(),
                    "com": d.xipos[i].tolist(),
                    "inertialQuat": quaternion(d.ximat[i].reshape(3, 3)).tolist(),
                }
                for i in range(m.nbody)
            ],
            "joints": [
                {
                    "name": m.joint(i).name,
                    "pos": d.xanchor[i].tolist(),
                    "axis": d.xaxis[i].tolist(),
                }
                for i in range(m.njnt)
            ],
            "sites": [
                {
                    "name": m.site(i).name,
                    "pos": d.site_xpos[i].tolist(),
                    "quat": quaternion(d.site_xmat[i].reshape(3, 3)).tolist(),
                }
                for i in range(m.nsite)
            ],
            "tendons": [
                [
                    d.wrap_xpos[j].tolist()
                    for j in range(
                        d.ten_wrapadr[i], d.ten_wrapadr[i] + d.ten_wrapnum[i]
                    )
                ]
                for i in range(m.ntendon)
            ],
        }

    def simulation(self, action, values=None):
        if self.model is None:
            raise ValueError("草稿尚未编译成功")
        values = values or {}
        if action == "align_joints":
            from .kinematics import align_joints
            self.running = False
            result = align_joints(self.model, self.data, values["first"], values["second"], values.get("frame", "world"), values.get("drivers"))
            return {**self.state(), "projection": result}
        if action == "project_joint":
            from .kinematics import project_joint

            self.running = False
            result = project_joint(
                self.model, self.data, values["name"], float(values["value"])
            )
            return {**self.state(), "projection": result}
        if action in ("joint", "ctrl") and not np.isfinite(float(values["value"])):
            raise ValueError("关节位置和控制量必须是有限数")
        if action == "play":
            if self.meta["stale"]:
                raise ValueError("请先确认改变几何的刚体惯性")
            self.running = True
        elif action == "pause":
            self.running = False
        elif action == "reset":
            self.running = False
            mj.mj_resetData(self.model, self.data)
        elif action == "joint":
            self.running = False
            jid = self.model.joint(values["name"]).id
            if self.model.jnt_type[jid] not in (2, 3):
                raise ValueError("滑块只支持标量关节")
            self.data.qpos[self.model.jnt_qposadr[jid]] = float(values["value"])
        elif action == "ctrl":
            self.data.ctrl[self.model.actuator(values["name"]).id] = float(
                values["value"]
            )
        elif action == "key":
            self.running = False
            mj.mj_resetDataKeyframe(
                self.model, self.data, self.model.key(values["name"]).id
            )
        else:
            raise ValueError("未知仿真操作")
        mj.mj_forward(self.model, self.data)
        return self.state()

    def archive(self, draft=True):
        output = io.BytesIO()
        if draft:
            xml = self.base
            assets = dict(self.base_assets)
            payload = {"version": 1, "journal": self.journal}
        else:
            if self.error or self.meta["stale"]:
                raise ValueError("请解决编译错误并确认所有过期惯性后导出")
            m = self.spec.compile()
            d = mj.MjData(m)
            for _ in range(100):
                mj.mj_step(m, d)
            if (
                not np.isfinite(d.qpos).all()
                or not np.isfinite(d.qvel).all()
                or any(w.number for w in d.warning)
            ):
                raise ValueError("导出模型短程仿真不稳定")
            from .persistence import readable_assets
            xml, assets = readable_assets(self.spec.to_xml(), dict(self.spec.assets))
            xml = xml.decode()
            payload = None
            rebuilt = mj.MjSpec.from_string(xml, assets=assets).compile()
            if (m.nq, m.nv, m.nu, m.ngeom) != (
                rebuilt.nq,
                rebuilt.nv,
                rebuilt.nu,
                rebuilt.ngeom,
            ):
                raise ValueError("导出重载结构不一致")
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("model.xml", xml)
            for name, content in assets.items():
                z.writestr(name, content)
            if payload is not None:
                z.writestr("editor.json", json.dumps(payload, ensure_ascii=False))
        return output.getvalue()

    @classmethod
    def from_archive(cls, content):
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            if (
                len(z.infolist()) > 10000
                or sum(v.file_size for v in z.infolist()) > 512 * 1024 * 1024
            ):
                raise ValueError("工程包过大")
            for entry in z.infolist():
                path = Path(entry.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("工程包含不安全路径")
            with tempfile.TemporaryDirectory(dir=RUNTIME) as folder:
                z.extractall(folder)
                root = Path(folder)
                xml = root / "model.xml"
                if not xml.exists():
                    xmls = list(root.rglob("*.xml"))
                    if len(xmls) != 1:
                        raise ValueError("多 XML 工程请将主入口命名为 model.xml")
                    xml = xmls[0]
                editor = cls(load_spec(xml, root))
                if "editor.json" in z.namelist():
                    payload = json.loads(z.read("editor.json"))
                    if payload.get("version") != 1:
                        raise ValueError("不支持的工程版本")
                    for command in payload["journal"]:
                        editor.command(editor.revision, **command)
                return editor
