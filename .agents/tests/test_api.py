import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app


def test_obj_mtl_texture_import_append_and_missing_resource():
    image = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 80, 20)).save(image, format="PNG")
    obj = """mtllib surface.mtl
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
vt 0 0
vt 1 0
vt 0 1
usemtl paint
f 1/1 3/3 2/2
f 1/1 2/2 4/3
usemtl plain
f 1/1 4/2 3/3
f 2/1 3/2 4/3
"""
    mtl = (
        "newmtl paint\nKd 0.8 0.4 0.2\nmap_Kd color.png\nnewmtl plain\nKd 0.3 0.4 0.5\n"
    )
    with TestClient(app) as client:
        scene = client.post("/api/example/empty").json()
        files = [
            ("files", ("part.obj", obj, "text/plain")),
            ("files", ("surface.mtl", mtl, "text/plain")),
        ]
        missing = client.post("/api/import", files=files, data={"scale": ".001"})
        assert missing.status_code == 422 and "color.png" in missing.text
        response = client.post(
            "/api/import",
            files=[*files, ("files", ("color.png", image.getvalue(), "image/png"))],
            data={"scale": ".001", "mode": "append", "revision": scene["revision"]},
        )
        assert response.status_code == 200, response.text
        scene = response.json()
        parts = [b for b in scene["bodies"] if b["name"] != "world"]
        assert len(parts) == 1
        geoms = [g for g in scene["geoms"] if g["parent"] == parts[0]["name"]]
        assert len(geoms) == 2
        assert any(g.get("texture") for g in geoms)
        assert (
            max(
                max(abs(v) for point in g["geometry"]["vertices"] for v in point)
                for g in geoms
            )
            < 0.002
        )
        stale = client.post(
            "/api/command", json={"revision": scene["revision"] - 1, "op": "undo"}
        )
        assert stale.status_code == 409
        result = client.post(
            "/api/command",
            json={
                "revision": scene["revision"],
                "op": "inertia",
                "target": parts[0]["id"],
                "args": {
                    "mode": "density",
                    "density": 1000,
                    "geoms": [g["name"] for g in geoms],
                },
            },
        )
        assert result.status_code == 200, result.text
        scene = result.json()
        assert not scene["meta"]["stale"]
        mass = next(b["mass"] for b in scene["bodies"] if b["name"] == parts[0]["name"])
        np.testing.assert_allclose(mass, 1e-9 * 1000 / 6, rtol=1e-5)
        archive = client.get("/api/archive?draft=false")
        assert archive.status_code == 200, archive.text
        reopened = client.post(
            "/api/import",
            files={"files": ("model.zip", archive.content, "application/zip")},
        )
        assert reopened.status_code == 200, reopened.text
        assert len(reopened.json()["geoms"]) == 3


def test_api_rejects_nonfinite_simulation_and_reports_progress():
    with TestClient(app) as client:
        scene = client.post("/api/example/complex").json()
        response = client.post(
            "/api/simulation",
            json={"action": "joint", "values": {"name": "gimbal_x", "value": "nan"}},
        )
        assert response.status_code == 422
        with client.websocket_connect("/api/live") as websocket:
            message = websocket.receive_json()
            assert message["revision"] == scene["revision"]
            assert message["state"]["valid"] and "progress" in message


def test_joint_projection_api_checks_revision_and_returns_feasibility():
    with TestClient(app) as client:
        scene = client.post("/api/example/complex").json()
        payload = {
            "action": "project_joint",
            "revision": scene["revision"],
            "values": {"name": "left_hinge", "value": 0.3},
        }
        response = client.post("/api/simulation", json=payload)
        assert response.status_code == 200
        assert response.json()["projection"]["accepted"]
        assert response.json()["constraintResidual"] < 1e-6
        payload["revision"] -= 1
        assert client.post("/api/simulation", json=payload).status_code == 422


def test_zero_export_roundtrip_and_stale_export_rejection():
    import zipfile
    from lxml import etree
    with TestClient(app) as client:
        scene = client.post('/api/example/complex').json()
        revision = scene['revision']
        calibrated = client.post('/api/command', json={
            'revision':revision, 'op':'zero', 'target':'joint:gimbal_x',
            'args':{'offset':-0.08443473357301605,'resolved':{}}})
        assert calibrated.status_code == 200, calibrated.text
        revision = calibrated.json()['revision']
        for draft in ['true','false']:
            stale = client.get(f'/api/archive?draft={draft}&revision={revision-1}')
            assert stale.status_code == 422 and '已阻止导出' in stale.text
        exported = client.get(f'/api/archive?draft=false&revision={revision}')
        assert exported.status_code == 200, exported.text
        xml = etree.fromstring(zipfile.ZipFile(io.BytesIO(exported.content)).read('model.xml'))
        assert abs(float(xml.find('.//joint[@name="gimbal_x"]').get('ref')) - 0.08443473357301605) < 1e-7
        reopened = client.post('/api/import',files={'files':('export.zip',exported.content,'application/zip')})
        assert reopened.status_code == 200,reopened.text
        joint = next(j for j in reopened.json()['joints'] if j['name']=='gimbal_x')
        assert abs(joint['fields']['ref']-0.08443473357301605) < 1e-7


def test_open_local_and_save_to_original(tmp_path):
    import mujoco as mj
    p=tmp_path/'robot.xml'
    p.write_text('<mujoco><worldbody><body><joint name="j"/><geom size=".1"/></body></worldbody></mujoco>')
    with TestClient(app) as client:
        rev=client.get('/api/scene').json()['revision']
        opened=client.post('/api/open-local',json={'revision':rev,'path':str(p)})
        assert opened.status_code==200,opened.text
        s=opened.json();assert s['sourcePath']==str(p)
        s=client.post('/api/command',json={'revision':s['revision'],'op':'zero','target':'joint:j','args':{'offset':.2}}).json()
        before=p.read_bytes()
        stale=client.post('/api/save',json={'revision':s['revision']-1})
        assert stale.status_code==422 and p.read_bytes()==before
        saved=client.post('/api/save',json={'revision':s['revision']})
        assert saved.status_code==200,saved.text
        assert abs(mj.MjModel.from_xml_path(str(p)).qpos0[0]+.2)<1e-6
