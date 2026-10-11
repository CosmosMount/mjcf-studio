import io
import zipfile
import numpy as np
import pytest
import mujoco as mj
import trimesh
from scipy.spatial.transform import Rotation
from backend.model import Editor, ROOT, load_spec
from backend.geometry import Mesh, combine, rotation, quaternion, valid_inertia


def box_editor():
    e = Editor()
    t = trimesh.creation.box(extents=[0.2, 0.4, 0.6])
    uv = np.arange(len(t.faces) * 6, dtype=float).reshape(-1, 3, 2) / 100
    e.command(
        0,
        "add_obj",
        args={
            "name": "part",
            "pos": [0, 0, 0.5],
            "mesh": Mesh(t.vertices, t.faces, uv).json(),
        },
    )
    return e


def physical(e):
    e.command(
        e.revision,
        "inertia",
        "body:part",
        {"mode": "density", "geoms": ["part_visual"], "density": 2700},
    )
    return e


def test_analytic_inertia_scaling_rotation_and_combination():
    t = trimesh.creation.box(extents=[0.2, 0.4, 0.6])
    mesh = Mesh(t.vertices, t.faces)
    p = mesh.mass_properties(2700)
    mass = 2700 * 0.2 * 0.4 * 0.6
    expected = mass / 12 * np.array([0.4**2 + 0.6**2, 0.2**2 + 0.6**2, 0.2**2 + 0.4**2])
    np.testing.assert_allclose(p["mass"], mass, rtol=1e-12)
    np.testing.assert_allclose(p["tensor"], np.diag(expected), rtol=1e-12, atol=1e-12)
    larger = Mesh(mesh.vertices * 3, mesh.faces).mass_properties(2700)
    np.testing.assert_allclose(
        larger["tensor"], np.asarray(p["tensor"]) * 3**5, rtol=1e-12
    )
    r = Rotation.from_euler("xyz", [0.3, 0.5, 0.7]).as_matrix()
    shifted = Mesh(mesh.vertices @ r.T + [1, 2, 3], mesh.faces).mass_properties(2700)
    np.testing.assert_allclose(shifted["center"], [1, 2, 3], atol=1e-12)
    np.testing.assert_allclose(
        shifted["tensor"], r @ np.diag(expected) @ r.T, atol=1e-9
    )
    p2 = {**p, "center": [1, 0, 0]}
    combined = combine([p, p2])
    np.testing.assert_allclose(combined["center"], [0.5, 0, 0])
    np.testing.assert_allclose(
        combined["tensor"], 2 * np.diag(expected) + np.diag([0, mass / 2, mass / 2])
    )


def test_split_faces_uv_and_cap_conservation():
    mesh = box_editor().geom_mesh("part_visual")
    ids = [i for i, f in enumerate(mesh.faces) if np.all(mesh.vertices[f, 0] > 0.09)]
    selected, remaining = mesh.split(ids)
    assert len(selected.faces) + len(remaining.faces) == len(mesh.faces)
    np.testing.assert_array_equal(selected.uv, mesh.uv[ids])
    assert not remaining.diagnostics()["valid_solid"]
    closed = remaining.cap()
    assert closed.diagnostics()["valid_solid"]
    np.testing.assert_allclose(closed.mass_properties(2700)["mass"], 129.6)
    np.testing.assert_array_equal(closed.uv[: len(remaining.faces)], remaining.uv)
    with pytest.raises(ValueError):
        selected.mass_properties()
    with pytest.raises(ValueError):
        mesh.split(list(range(len(mesh.faces))))


def test_mesh_failures_and_inertia_validation():
    box = trimesh.creation.box()
    open_mesh = Mesh(box.vertices, box.faces[:-1])
    assert not open_mesh.diagnostics()["watertight"]
    with pytest.raises(ValueError):
        open_mesh.mass_properties()
    inverted = Mesh(box.vertices, box.faces[:, ::-1])
    assert not inverted.diagnostics()["valid_solid"]
    duplicate = Mesh(box.vertices, np.vstack([box.faces, box.faces[:1]]))
    assert duplicate.diagnostics()["nonmanifold_edges"]
    crossed = Mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, -1], [0.2, 0.2, 1], [0.8, 0.2, 0]],
        [[0, 1, 2], [3, 4, 5]],
    )
    assert crossed.diagnostics()["self_intersections"]
    with pytest.raises(ValueError):
        valid_inertia(1, [0, 0, 0], np.diag([1, 1, 3]))
    with pytest.raises(ValueError):
        valid_inertia(-1, [0, 0, 0], np.eye(3))


def test_transactions_undo_redo_revision_and_draft():
    e = box_editor()
    rev = e.revision
    e.command(rev, "set", "geom:part_visual", {"rgba": [1, 0, 0, 1]})
    with pytest.raises(ValueError):
        e.command(rev, "set", "geom:part_visual", {"rgba": [0, 1, 0, 1]})
    e.command(e.revision, "undo")
    np.testing.assert_allclose(e.spec.geom("part_visual").rgba, [0.5, 0.5, 0.5, 1])
    e.command(e.revision, "redo")
    np.testing.assert_allclose(e.spec.geom("part_visual").rgba, [1, 0, 0, 1])
    restored = Editor.from_archive(e.archive(True))
    np.testing.assert_allclose(restored.spec.geom("part_visual").rgba, [1, 0, 0, 1])
    with pytest.raises(ValueError):
        e.archive(False)
    with pytest.raises(ValueError):
        e.command(e.revision, "set", "geom:part_visual", {"whatever": 42})
    assert e.revision == rev + 3


def test_export_reload_mass_texture_and_mesh_coordinates():
    e = physical(box_editor())
    e.command(
        e.revision,
        "material",
        "geom:part_visual",
        {"texture": "checker", "rgba": [0.7, 0.5, 0.3, 1]},
    )
    s = e.scene()
    geom = next(g for g in s["geoms"] if g["name"] == "part_visual")
    assert geom.get("texture")
    world = (
        np.asarray(geom["geometry"]["vertices"]) @ rotation(geom["worldQuat"]).T
        + geom["worldPos"]
    )
    np.testing.assert_allclose(world.min(0), [-0.1, -0.2, 0.2], atol=1e-7)
    np.testing.assert_allclose(world.max(0), [0.1, 0.2, 0.8], atol=1e-7)
    output = Editor.from_archive(e.archive(False))
    np.testing.assert_allclose(output.model.body_mass, e.model.body_mass, rtol=1e-5)


def test_complex_roundtrip_parameters_motion_and_keyframe():
    e = Editor(load_spec(ROOT / "examples/complex.xml", ROOT / "examples"))
    other = Editor.from_archive(e.archive(False))
    for attr in [
        "body_mass",
        "body_inertia",
        "jnt_axis",
        "jnt_range",
        "eq_data",
        "actuator_gear",
        "key_qpos",
        "tendon_lengthspring",
    ]:
        np.testing.assert_allclose(
            getattr(e.model, attr), getattr(other.model, attr), atol=2e-6, rtol=2e-5
        )
    assert (
        e.model.neq,
        e.model.ntendon,
        e.model.nu,
        e.model.nsensor,
        e.model.nkey,
    ) == (2, 2, 2, 2, 1)
    for ed in (e, other):
        ed.data.ctrl[:] = [0.1, 0.03]
        for _ in range(100):
            mj.mj_step(ed.model, ed.data)
    np.testing.assert_allclose(e.data.qpos, other.data.qpos, atol=2e-6, rtol=2e-5)


def test_include_and_degrees_roundtrip(tmp_path):
    (tmp_path / "body.xml").write_text(
        '<mujocoinclude><body name="b"><joint name="j" ref="30" range="-45 60"/><geom name="g" type="box" size=".1 .2 .3"/></body></mujocoinclude>'
    )
    (tmp_path / "model.xml").write_text(
        '<mujoco><default><joint damping=".7"/></default><worldbody><include file="body.xml"/></worldbody></mujoco>'
    )
    e = Editor(load_spec(tmp_path / "model.xml", tmp_path))
    assert not e.spec.compiler.degree
    np.testing.assert_allclose(e.spec.joint("j").ref, np.pi / 6, rtol=2e-6)
    np.testing.assert_allclose(e.spec.joint("j").damping, [0.7, 0, 0])
    out = Editor.from_archive(e.archive(False))
    np.testing.assert_allclose(out.model.qpos0, e.model.qpos0, atol=1e-6)


def test_reframe_preserves_multiple_configurations_dynamics():
    spec = mj.MjSpec.from_string(
        """<mujoco><compiler angle="radian"/><worldbody><body name="b" pos=".2 .3 .4" quat=".9238795 0 .3826834 0"><joint name="j" axis="0 0 1" pos=".1 0 .2" damping=".1"/><geom name="g" type="box" size=".1 .2 .3" pos=".3 0 .2"/><site name="s" pos=".1 .2 .3"/><body name="child" pos="0 .2 .4"><geom name="c" type="sphere" size=".05"/></body></body></worldbody></mujoco>"""
    )
    e = Editor(spec)
    original = e.model
    e.command(
        e.revision,
        "reframe",
        "body:b",
        {
            "pos": [0.5, -0.1, 0.2],
            "quat": quaternion(
                Rotation.from_euler("xyz", [0.2, 0.5, 0.7]).as_matrix()
            ).tolist(),
        },
    )
    assert not e.error
    for q in [-0.6, 0, 0.7]:
        a, b = mj.MjData(original), mj.MjData(e.model)
        a.qpos[:] = b.qpos[:] = q
        a.qvel[:] = b.qvel[:] = 0.4
        mj.mj_forward(original, a)
        mj.mj_forward(e.model, b)
        np.testing.assert_allclose(a.geom_xpos, b.geom_xpos, atol=2e-6)
        np.testing.assert_allclose(a.geom_xmat, b.geom_xmat, atol=2e-6)
        np.testing.assert_allclose(a.site_xpos, b.site_xpos, atol=2e-6)
        np.testing.assert_allclose(a.xipos[1:], b.xipos[1:], atol=2e-6)
        np.testing.assert_allclose(a.qacc, b.qacc, atol=1e-5)


def test_zero_reference_spring_limits_keys_and_equality():
    s = mj.MjSpec.from_string(
        """<mujoco><compiler angle="radian"/><worldbody><body name="b"><joint name="j" ref=".2" springref=".1" stiffness="2" range="-1 2"/><geom type="box" size=".1 .2 .3"/></body><body name="c" pos="1 0 0"><joint name="k"/><geom type="box" size=".1 .2 .3"/></body></worldbody><equality><joint name="couple" joint1="j" joint2="k" polycoef="0 1 .1 0 0"/></equality><actuator><motor name="motor" joint="j"/></actuator><keyframe><key name="pose" qpos=".6 .4"/></keyframe></mujoco>"""
    )
    e = Editor(s)
    old = e.model
    e.command(e.revision, "zero", "joint:j", {"offset": 0.5})
    assert e.spec.joint("j").ref == pytest.approx(-0.3)
    np.testing.assert_allclose(e.spec.joint("j").range, [-1.5, 1.5])
    np.testing.assert_allclose(e.spec.key("pose").qpos, [0.1, 0.4])
    for q in [0.2, 0.6]:
        a, b = mj.MjData(old), mj.MjData(e.model)
        a.qpos[:] = [q, 0.1]
        b.qpos[:] = [q - 0.5, 0.1]
        a.ctrl[:] = b.ctrl[:] = 0.3
        mj.mj_forward(old, a)
        mj.mj_forward(e.model, b)
        np.testing.assert_allclose(a.geom_xpos, b.geom_xpos, atol=1e-10)
        np.testing.assert_allclose(a.qfrc_passive, b.qfrc_passive, atol=1e-10)
        np.testing.assert_allclose(a.efc_pos, b.efc_pos, atol=1e-10)
        np.testing.assert_allclose(a.qfrc_actuator, b.qfrc_actuator, atol=1e-10)


def test_coupled_zero_rejects_atomically_and_tendon_parameters():
    e = Editor(load_spec(ROOT / "examples/complex.xml", ROOT / "examples"))
    before = e.spec.to_xml()
    rev = e.revision
    with pytest.raises(ValueError, match="tendon:differential"):
        e.command(rev, "zero", "joint:left_hinge", {"offset": 0.5})
    assert e.spec.to_xml() == before and e.revision == rev
    e.command(
        e.revision,
        "replace_tendon",
        "tendon:differential",
        {
            "path": [
                {"type": "joint", "target": "left_hinge", "coef": 2},
                {"type": "joint", "target": "right_hinge", "coef": -2},
            ]
        },
    )
    assert e.spec.tendon("differential").path[0].coef == 2
    assert e.model.nu == 2 and e.model.nsensor == 2
    assert not e.error


def test_archive_rejects_path_escape():
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, "w") as z:
        z.writestr("../unsafe", "bad")
    with pytest.raises(ValueError, match="不安全"):
        Editor.from_archive(buff.getvalue())


def test_capped_articulated_split_mass_conservation_and_reload():
    e = Editor()
    box = trimesh.creation.box(extents=[0.4, 0.2, 0.6]).subdivide().subdivide()
    e.command(
        0,
        "add_obj",
        args={"name": "part", "mesh": Mesh(box.vertices, box.faces).json()},
    )
    mesh = e.geom_mesh("part_visual")
    indices = np.flatnonzero(mesh.vertices[mesh.faces].mean(axis=1)[:, 2] > 0).tolist()
    before = mesh.mass_properties(1000)
    e.command(
        e.revision,
        "split",
        "geom:part_visual",
        {"faces": indices, "cap": True, "name": "segment"},
    )
    assert e.error is None and e.model.njnt == 1
    for name, geom in [("part", "part_visual"), ("segment", "part_visual_split")]:
        e.command(
            e.revision,
            "inertia",
            "body:" + name,
            {"mode": "density", "density": 1000, "geoms": [geom]},
        )
    np.testing.assert_allclose(e.model.body_mass.sum(), before["mass"], rtol=1e-12)
    assert not e.meta["stale"]
    result = Editor.from_archive(e.archive(False))
    assert result.model.njnt == 1
    np.testing.assert_allclose(result.model.body_mass, e.model.body_mass, rtol=1e-6)


def test_native_primitives_and_material_density():
    from backend.geometry import primitive_properties

    for kind, size in [
        (2, [0.2, 0, 0]),
        (3, [0.2, 0.3, 0]),
        (4, [0.2, 0.3, 0.4]),
        (5, [0.2, 0.3, 0]),
        (6, [0.2, 0.3, 0.4]),
    ]:
        spec = mj.MjSpec()
        b = spec.worldbody.add_body(name="b")
        b.add_geom(name="g", type=kind, size=size, density=1000)
        model = spec.compile()
        p = primitive_properties(kind, size, 1000)
        np.testing.assert_allclose(p["mass"], model.body_mass[1], rtol=1e-12)
        np.testing.assert_allclose(
            np.sort(np.linalg.eigvalsh(p["tensor"])),
            np.sort(model.body_inertia[1]),
            rtol=1e-12,
        )
    e = box_editor()
    e.command(e.revision, "material", "geom:part_visual", {"density": 2700})
    e.command(
        e.revision,
        "inertia",
        "body:part",
        {"mode": "materials", "geoms": ["part_visual"]},
    )
    assert e.spec.body("part").mass == pytest.approx(129.6)


def test_append_keeps_existing_and_all_relations():
    e = Editor()
    other = Editor(load_spec(ROOT / "examples/complex.xml", ROOT / "examples"))
    e.command(
        0,
        "attach_model",
        "",
        {
            "xml": other.spec.to_xml(),
            "assets": {},
            "prefix": "added_",
            "meta": other.meta,
        },
    )
    assert e.spec.geom("ground") is not None
    assert (
        e.model.njnt,
        e.model.neq,
        e.model.ntendon,
        e.model.nu,
        e.model.nsensor,
        e.model.nkey,
    ) == (5, 2, 2, 2, 2, 1)
    assert any(
        r["references"] == ["joint:added_left_hinge", "joint:added_right_hinge"]
        for r in e.relations()
    )
    e.command(e.revision, "undo")
    assert e.model.njnt == 0


def test_invalid_draft_persists_and_rolls_back():
    e = box_editor()
    e.command(e.revision, "add_joint", "", {"body": "part", "name": "joint"})
    # Invalid dimensions are a recoverable draft, not a successful export.
    e.command(e.revision, "set", "joint:joint", {"range": [2, -1]})
    assert e.error and not e.scene()["state"]["valid"]
    restored = Editor.from_archive(e.archive(True))
    assert restored.error
    with pytest.raises(ValueError):
        e.archive(False)
    restored.command(restored.revision, "undo")
    assert not restored.error


def test_external_include_is_rejected_before_parsing(tmp_path):
    path = tmp_path / "model.xml"
    path.write_text('<mujoco><include file="/etc/passwd"/></mujoco>')
    with pytest.raises(ValueError, match="工程目录外"):
        load_spec(path, tmp_path)


@pytest.mark.parametrize("joint_type", ["free", "ball"])
def test_quaternion_joint_reframe_preserves_keyframe_world_geometry_and_energy(
    joint_type,
):
    q = (
        "1 2 3 .9238795 .3826834 0 0"
        if joint_type == "free"
        else ".9238795 .3826834 0 0"
    )
    v = ".1 .2 .3 .4 .5 .6" if joint_type == "free" else ".4 .5 .6"
    spec = mj.MjSpec.from_string(
        f'<mujoco><worldbody><body name="b" pos=".2 .3 .4"><joint name="j" type="{joint_type}"/><geom name="g" type="box" size=".1 .2 .3" pos=".2 0 .3"/><site name="tip" pos=".3 .2 .1"/></body></worldbody><keyframe><key name="pose" qpos="{q}" qvel="{v}"/></keyframe></mujoco>'
    )
    e = Editor(spec)
    old = e.model
    a = mj.MjData(old)
    mj.mj_resetDataKeyframe(old, a, 0)
    mj.mj_forward(old, a)
    mj.mj_energyVel(old, a)
    e.command(
        e.revision,
        "reframe",
        "body:b",
        {
            "pos": [0.4, 0.1, 0.5],
            "quat": quaternion(
                Rotation.from_euler("xyz", [0.2, 0.3, 0.4]).as_matrix()
            ).tolist(),
        },
    )
    assert not e.error
    b = mj.MjData(e.model)
    mj.mj_resetDataKeyframe(e.model, b, 0)
    mj.mj_forward(e.model, b)
    mj.mj_energyVel(e.model, b)
    np.testing.assert_allclose(a.geom_xpos, b.geom_xpos, atol=2e-6)
    np.testing.assert_allclose(a.geom_xmat, b.geom_xmat, atol=2e-6)
    np.testing.assert_allclose(a.site_xpos, b.site_xpos, atol=2e-6)
    np.testing.assert_allclose(a.energy[1], b.energy[1], rtol=2e-6)


def test_reframe_multijoint_body_and_closed_loop():
    e = Editor(load_spec(ROOT / "examples/complex.xml", ROOT / "examples"))
    old = e.model
    e.command(
        e.revision,
        "reframe",
        "body:gimbal",
        {
            "pos": [0.8, 0.1, 0.4],
            "quat": quaternion(
                Rotation.from_euler("xyz", [0.1, 0.3, 0.2]).as_matrix()
            ).tolist(),
        },
    )
    for q in ([0, 0, 0, 0.2, 0.3], [0.1, -0.1, 0.1, -0.4, 0.2]):
        a, b = mj.MjData(old), mj.MjData(e.model)
        a.qpos[:] = b.qpos[:] = q
        mj.mj_forward(old, a)
        mj.mj_forward(e.model, b)
        np.testing.assert_allclose(a.geom_xpos, b.geom_xpos, atol=2e-6)
        np.testing.assert_allclose(a.geom_xmat, b.geom_xmat, atol=2e-6)
        np.testing.assert_allclose(a.efc_pos, b.efc_pos, atol=2e-6)


def test_cap_material_retains_mass_and_compound_surface_calculation():
    e = Editor()
    box = trimesh.creation.box().subdivide()
    e.command(
        0,
        "add_obj",
        args={"name": "part", "mesh": Mesh(box.vertices, box.faces).json()},
    )
    mesh = e.geom_mesh("part_visual")
    ids = np.flatnonzero(mesh.vertices[mesh.faces].mean(1)[:, 2] > 0).tolist()
    e.command(
        e.revision,
        "split",
        "geom:part_visual",
        {"faces": ids, "cap": True, "name": "segment"},
    )
    for body, geom in [("part", "part_visual"), ("segment", "part_visual_split")]:
        e.command(
            e.revision,
            "inertia",
            "body:" + body,
            {"mode": "density", "density": 1000, "geoms": [geom]},
        )
    before = e.model.body_mass.copy()
    e.command(
        e.revision, "cap_material", "geom:part_visual_split", {"rgba": [1, 0, 0, 1]}
    )
    assert not e.error
    np.testing.assert_allclose(e.model.body_mass, before)
    assert e.spec.geom("part_visual_split_cap").contype == 0
    e.command(
        e.revision,
        "inertia",
        "body:segment",
        {
            "mode": "density",
            "density": 1000,
            "geoms": ["part_visual_split", "part_visual_split_cap"],
        },
    )
    np.testing.assert_allclose(e.model.body_mass, before)
    assert Editor.from_archive(e.archive(False)).model.njnt == 1


def test_added_joint_remaps_keyframes_and_split_moves_dependencies():
    e = Editor(load_spec(ROOT / "examples/complex.xml", ROOT / "examples"))
    old = e.model
    e.command(
        e.revision,
        "add_joint",
        "",
        {"body": "gimbal", "name": "extra", "type": "slide"},
    )
    assert e.model.nq == old.nq + 1 and len(e.spec.key("home").qpos) == e.model.nq
    assert e.model.nu == old.nu and e.model.nsensor == old.nsensor


def test_split_reparents_subtree_site_and_preserves_sensor_and_keyframe():
    e = Editor()
    box = trimesh.creation.box().subdivide()
    e.command(
        0,
        "add_obj",
        args={"name": "part", "mesh": Mesh(box.vertices, box.faces).json()},
    )
    parent = e.spec.body("part")
    parent.add_site(name="marker", pos=[0.1, 0.2, 0.3])
    child = parent.add_body(name="child", pos=[0.2, 0, 0.2])
    child.add_joint(name="child_j", type=mj.mjtJoint.mjJNT_SLIDE)
    child.add_geom(name="child_g", type=mj.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
    e.spec.add_sensor(
        name="marker_position",
        type=mj.mjtSensor.mjSENS_FRAMEPOS,
        objtype=mj.mjtObj.mjOBJ_SITE,
        objname="marker",
    )
    key = e.spec.add_key(name="pose")
    key.qpos = [0.3]
    e = Editor(e.spec)
    old = e.model
    m = e.geom_mesh("part_visual")
    ids = np.flatnonzero(m.vertices[m.faces].mean(1)[:, 2] > 0).tolist()
    e.command(
        e.revision,
        "split",
        "geom:part_visual",
        {
            "faces": ids,
            "cap": True,
            "name": "segment",
            "move": ["body:child", "site:marker"],
        },
    )
    assert not e.error
    assert (
        e.spec.body("child").parent.name == "segment"
        and e.spec.site("marker").parent.name == "segment"
    )
    assert e.model.nsensor == 1 and e.model.nq == 2
    a, b = mj.MjData(old), mj.MjData(e.model)
    mj.mj_resetDataKeyframe(old, a, 0)
    mj.mj_resetDataKeyframe(e.model, b, 0)
    mj.mj_forward(old, a)
    mj.mj_forward(e.model, b)
    np.testing.assert_allclose(
        a.geom_xpos[old.geom("child_g").id],
        b.geom_xpos[e.model.geom("child_g").id],
        atol=1e-6,
    )
    np.testing.assert_allclose(a.sensordata, b.sensordata, atol=1e-6)


def test_generated_resource_retirement_preserves_undo():
    e = physical(box_editor())
    e.command(e.revision, "material", "geom:part_visual", {"texture": "checker"})
    old = e.spec.geom("part_visual").material
    e.command(
        e.revision,
        "material",
        "geom:part_visual",
        {"texture": "keep", "rgba": [1, 0, 0, 1]},
    )
    assert len(e.spec.materials) == 1 and len(e.spec.textures) == 1
    assert e.spec.geom("part_visual").material != old
    e.command(e.revision, "undo")
    assert e.spec.geom("part_visual").material == old and e.model.ntex == 1


def test_zero_keeps_current_preview_pose():
    e = Editor(load_spec(ROOT / "examples/complex.xml", ROOT / "examples"))
    e.simulation("joint", {"name": "gimbal_x", "value": 0.8})
    positions = e.data.geom_xpos.copy()
    rotations = e.data.geom_xmat.copy()
    e.command(e.revision, "zero", "joint:gimbal_x", {"offset": 0.8})
    assert e.data.qpos[
        e.model.jnt_qposadr[e.model.joint("gimbal_x").id]
    ] == pytest.approx(0)
    np.testing.assert_allclose(e.data.geom_xpos, positions, atol=1e-12)
    np.testing.assert_allclose(e.data.geom_xmat, rotations, atol=1e-12)


def test_closed_chain_joint_projection_tracks_and_preserves_unrelated_joints():
    e = Editor(load_spec(ROOT / "examples/complex.xml", ROOT / "examples"))
    e.simulation("joint", {"name": "gimbal_x", "value": 0.4})
    revision, xml = e.revision, e.spec.to_xml()
    for value in [0.1, 0.4, 0.8, -0.4, 0.0]:
        state = e.simulation("project_joint", {"name": "left_hinge", "value": value})
        assert state["projection"]["accepted"]
        assert state["constraintResidual"] < 1e-6
        np.testing.assert_allclose(
            e.data.site("coupler_tip").xpos, e.data.site("right_tip").xpos, atol=1e-6
        )
        assert e.data.joint("left_hinge").qpos[0] == pytest.approx(value)
        assert e.data.joint("right_hinge").qpos[0] == pytest.approx(value, abs=1e-6)
        assert e.data.joint("gimbal_x").qpos[0] == pytest.approx(0.4)
    assert e.revision == revision and e.spec.to_xml() == xml


def test_project_joint_failure_is_atomic_and_scalar_limits_are_enforced():
    e = Editor(
        mj.MjSpec.from_string(
            """<mujoco><worldbody><body><freejoint/><geom size=".1"/><body><joint name="drive" range="-.2 .2"/><geom size=".05"/></body></body></worldbody><equality><joint joint1="drive" polycoef="0 0 0 0 0"/></equality></mujoco>"""
        )
    )
    e.data.qvel[:] = 0.1
    original = e.data.qpos.copy(), e.data.qvel.copy()
    state = e.simulation("project_joint", {"name": "drive", "value": 0.1})
    assert not state["projection"]["accepted"]
    np.testing.assert_array_equal(e.data.qpos, original[0])
    np.testing.assert_array_equal(e.data.qvel, original[1])
    e.data.eq_active[:] = False
    state = e.simulation("project_joint", {"name": "drive", "value": 1})
    assert state["projection"]["accepted"] and state["projection"][
        "value"
    ] == pytest.approx(np.deg2rad(0.2))
    np.testing.assert_array_equal(e.data.qpos[:7], original[0][:7])
    with pytest.raises(ValueError, match="有限"):
        e.simulation("project_joint", {"name": "drive", "value": float("nan")})


def test_projection_respects_tendon_travel_and_sparse_constraints():
    e = Editor(
        mj.MjSpec.from_string(
            """<mujoco><compiler angle="radian"/><option jacobian="sparse"/><worldbody><body><joint name="drive" type="slide" range="-1 1"/><geom size=".05"/></body><body><joint name="follow" type="slide" range="-.1 .1"/><geom size=".05"/></body></worldbody><tendon><fixed name="cable" limited="true" range=".05 .1"><joint joint="drive" coef="1"/><joint joint="follow" coef="1"/></fixed></tendon></mujoco>"""
        )
    )
    state = e.simulation("project_joint", {"name": "drive", "value": 0.15})
    assert state["projection"]["accepted"]
    assert 0.05 - 1e-6 <= e.data.ten_length[0] <= 0.1 + 1e-6
    original = e.data.qpos.copy()
    state = e.simulation("project_joint", {"name": "drive", "value": 0.4})
    assert not state["projection"]["accepted"]
    np.testing.assert_array_equal(original, e.data.qpos)


def test_joint_alignment_and_reference_roundtrip():
    from backend.kinematics import align_joints
    xml = '''<mujoco><worldbody><body name="arm"><joint name="pivot" axis="0 1 0"/><geom type="capsule" size=".05" fromto="0 0 0 1 0 1"/><body pos="1 0 1"><joint name="tip" axis="0 1 0"/><geom size=".1"/></body></body></worldbody></mujoco>'''
    e = Editor(mj.MjSpec.from_string(xml))
    original_distance = np.linalg.norm(e.data.xanchor[1] - e.data.xanchor[0])
    result = e.simulation('align_joints', {'first':'pivot', 'second':'tip', 'drivers':['pivot']})
    assert result['projection']['accepted'] is True
    assert abs(e.data.xanchor[1, 2] - e.data.xanchor[0, 2]) < 1e-6
    assert np.linalg.norm(e.data.xanchor[1] - e.data.xanchor[0]) == pytest.approx(original_distance)
    q = e.data.qpos.copy()
    # In arm coordinates the line remains sloped regardless of pivot rotation.
    failure = align_joints(e.model, e.data, 'pivot', 'tip', 'arm', ['pivot'])
    assert not failure['accepted']
    np.testing.assert_array_equal(e.data.qpos, q)
    e.command(e.revision, 'save_pose', args={'name':'horizontal', 'qpos':q.tolist()})
    np.testing.assert_allclose(e.data.qpos, q)
    e.command(e.revision, 'zero', 'joint:pivot', {'offset':float(q[0])})
    assert abs(e.data.qpos[0]) < 1e-12
    assert abs(e.data.xanchor[1, 2] - e.data.xanchor[0, 2]) < 1e-6
    reopened = Editor.from_archive(e.archive())
    assert abs(reopened.model.key('horizontal').qpos[0]) < 1e-12
    e.command(e.revision, 'undo')
    np.testing.assert_allclose(e.model.key('horizontal').qpos, q)
    e.command(e.revision, 'undo')
    assert e.model.nkey == 0


def test_joint_alignment_rejects_coincident_and_joint_limits():
    from backend.kinematics import align_joints
    s = mj.MjSpec.from_string('''<mujoco><compiler angle="radian"/><worldbody><body><joint name="a" axis="0 1 0" limited="true" range="-.1 .1"/><geom size=".1"/><body pos="1 0 1"><joint name="b"/><geom size=".1"/></body></body></worldbody></mujoco>''')
    e = Editor(s); q=e.data.qpos.copy()
    assert not align_joints(e.model,e.data,'a','b',drivers=['a'])['accepted']
    np.testing.assert_array_equal(e.data.qpos,q)
    with pytest.raises(ValueError, match='不同'):
        align_joints(e.model,e.data,'a','a',drivers=['a'])


def test_local_save_zero_assets_reopen_and_conflict(tmp_path):
    from backend.persistence import bind_source, save_local
    e = physical(box_editor())
    e.command(e.revision, 'add_joint', args={'body':'part','name':'hinge','type':'hinge'})
    e.command(e.revision,'zero','joint:hinge',{'offset':.25})
    path=tmp_path/'original.xml'
    result=save_local(e,str(path))
    assert result['path']==str(path)
    m=mj.MjModel.from_xml_path(str(path))
    assert m.qpos0[m.joint('hinge').qposadr][0]==pytest.approx(-.25)
    assert list((tmp_path/'assets').iterdir())
    e.command(e.revision,'zero','joint:hinge',{'offset':.1})
    save_local(e)
    m=mj.MjModel.from_xml_path(str(path))
    assert m.qpos0[m.joint('hinge').qposadr][0]==pytest.approx(-.35)
    path.write_text(path.read_text()+'\n<!-- external edit -->\n')
    external=path.read_bytes()
    with pytest.raises(ValueError,match='外部修改'):
        save_local(e)
    assert path.read_bytes()==external
    project=tmp_path/'robot.mjcfproj'
    save_local(e,str(project))
    reopened=Editor.from_archive(project.read_bytes())
    assert reopened.spec.joint('hinge').ref==pytest.approx(-.35)
    assert reopened.source_path is None  # Uploaded files never grant a disk destination.


def test_readable_asset_names_reuse_after_color_and_avoid_overwrite(tmp_path):
    from backend.persistence import save_local
    e=physical(box_editor())
    e.command(e.revision,'add_joint',args={'body':'part','name':'hinge','type':'hinge'})
    path=tmp_path/'robot.xml'
    save_local(e,str(path))
    assets=list((tmp_path/'assets').iterdir())
    assert len(assets)==1 and assets[0].stem=='hinge_part'
    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in assets}
    e.command(e.revision,'set','geom:part_visual',{'rgba':[.2,.3,.4,1]})
    save_local(e)
    assert {p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in (tmp_path/'assets').iterdir()}==before
    assert mj.MjModel.from_xml_path(str(path)).geom('part_visual').rgba[0]==pytest.approx(.2)
    reopened=Editor(load_spec(path,tmp_path));save_local(reopened,str(path))
    assert {p.name for p in (tmp_path/'assets').iterdir()}==set(before)
    # A preexisting different asset must remain intact even when names collide.
    assets[0].write_bytes(b'external-resource')
    save_local(e)
    assert assets[0].read_bytes()==b'external-resource'
    assert (tmp_path/'assets'/'hinge_part_2.obj').exists()
    mj.MjModel.from_xml_path(str(path))
    exported=zipfile.ZipFile(io.BytesIO(e.archive(False)))
    assert 'assets/hinge_part.obj' in exported.namelist()
