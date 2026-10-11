from __future__ import annotations
import asyncio
import base64
import io
import os

os.environ.setdefault("MUJOCO_GL", "egl")
from pathlib import Path
import tempfile
import threading
import time
import shlex
from contextlib import asynccontextmanager
import numpy as np
import mujoco as mj
import trimesh
from PIL import Image
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .model import Editor, ROOT, RUNTIME, load_spec
from .geometry import Mesh


@asynccontextmanager
async def lifespan(application):
    task = asyncio.create_task(simulation_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="MJCF Studio", version="0.1.0", lifespan=lifespan)
editor = Editor()
lock = threading.RLock()
progress = {"active": False, "stage": "就绪"}
last_snapshot = {"revision": editor.revision, "state": editor.state()}


def validate_obj_resources(path, root):
    def resource(parent, name):
        candidate = (parent / name).resolve()
        if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
            raise ValueError("OBJ/MTL 资源缺失或位于工程外: " + name)
        return candidate

    options = {
        "-o": 3,
        "-s": 3,
        "-t": 3,
        "-mm": 2,
        "-bm": 1,
        "-clamp": 1,
        "-blendu": 1,
        "-blendv": 1,
        "-imfchan": 1,
        "-texres": 1,
        "-type": 1,
    }
    for line in path.read_text(errors="strict").splitlines():
        tokens = shlex.split(line, comments=True)
        if not tokens or tokens[0] != "mtllib":
            continue
        for name in tokens[1:]:
            material = resource(path.parent, name)
            for definition in material.read_text(errors="strict").splitlines():
                tokens = shlex.split(definition, comments=True)
                if not tokens or not (
                    tokens[0].startswith("map_")
                    or tokens[0] in ("bump", "disp", "decal", "refl")
                ):
                    continue
                tokens = tokens[1:]
                while tokens and tokens[0].startswith("-"):
                    option = tokens.pop(0)
                    if option not in options:
                        raise ValueError("不支持的 MTL 贴图选项: " + option)
                    count = options[option]
                    if option in ("-o", "-s", "-t"):
                        for _ in range(count):
                            try:
                                float(tokens[0])
                            except (ValueError, IndexError):
                                break
                            tokens.pop(0)
                    else:
                        tokens = tokens[count:]
                if not tokens:
                    raise ValueError("MTL 贴图缺少文件名")
                resource(material.parent, " ".join(tokens))


class Command(BaseModel):
    revision: int
    op: str
    target: str = ""
    args: dict = Field(default_factory=dict)


def guarded(operation):
    global progress
    with lock:
        progress = {
            "active": True,
            "stage": "处理模型与资源",
            "started": time.monotonic(),
        }
        try:
            return operation()
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        finally:
            progress = {"active": False, "stage": "就绪"}


@app.get("/api/scene")
def scene():
    return guarded(lambda: editor.scene())


@app.post("/api/open-local")
def open_local(request: dict):
    def operation():
        global editor
        from .persistence import destination, bind_source
        if request.get("revision") != editor.revision:
            raise ValueError("模型已更新，请刷新后重试")
        path = destination(request["path"])
        replacement = (Editor.from_archive(path.read_bytes()) if path.suffix.lower() == ".mjcfproj"
                       else Editor(load_spec(path, path.parent)))
        if replacement.error:
            raise ValueError(replacement.error)
        bind_source(replacement, path)
        replacement.revision = editor.revision + 1
        editor = replacement
        return editor.scene()
    return guarded(operation)


@app.post("/api/save")
def save(request: dict):
    def operation():
        from .persistence import save_local
        if request.get("revision") != editor.revision:
            raise ValueError("页面模型与服务端不一致，已阻止保存；请刷新并核对标定")
        return save_local(editor, request.get("path"))
    return guarded(operation)


@app.post("/api/command")
def command(request: Command):
    if request.revision != editor.revision:
        raise HTTPException(409, "模型修订号过期，请刷新")
    return guarded(lambda: editor.command(**request.model_dump()))


@app.post("/api/example/{name}")
def example(name: str):
    global editor

    def operation():
        global editor
        if name == "complex":
            replacement = Editor(
                load_spec(ROOT / "examples/complex.xml", ROOT / "examples")
            )
        elif name == "obj":
            replacement = Editor()
            mesh = trimesh.load_mesh(
                ROOT / "examples/split.obj", process=False, maintain_order=True
            )
            face_uv = mesh.visual.uv[mesh.faces]
            replacement.command(
                0,
                "add_obj",
                args={
                    "name": "sample",
                    "scale": 1,
                    "pos": [0, 0, 0.4],
                    "mesh": Mesh(mesh.vertices, mesh.faces, face_uv).json(),
                },
            )
        elif name == "empty":
            replacement = Editor()
        else:
            raise ValueError("未知示例")
        replacement.revision = editor.revision + 1
        editor = replacement
        return editor.scene()

    return guarded(operation)


@app.post("/api/import")
async def import_files(
    files: list[UploadFile] = File(...),
    entry: str = Form(""),
    scale: float = Form(1),
    mode: str = Form("replace"),
    revision: int = Form(-1),
):
    global editor
    contents = []
    total = 0
    for file in files:
        raw = await file.read()
        total += len(raw)
        if total > 512 * 1024 * 1024:
            raise HTTPException(413, "资源总大小超过 512 MiB")
        filename = (file.filename or "unnamed").replace("\\", "/")
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise HTTPException(422, "不安全文件路径")
        contents.append((filename, raw))

    def operation():
        global editor
        if len(contents) == 1 and contents[0][0].lower().endswith(
            (".zip", ".mjcfproj", ".mjz")
        ):
            replacement = Editor.from_archive(contents[0][1])
        else:
            with tempfile.TemporaryDirectory(dir=RUNTIME) as folder:
                root = Path(folder)
                for filename, raw in contents:
                    dest = root / filename
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw)
                names = [name for name, _ in contents]
                chosen = entry or next(
                    (n for n in names if Path(n).name == "model.xml"), ""
                )
                if not chosen:
                    candidates = [
                        n for n in names if n.lower().endswith((".xml", ".obj"))
                    ]
                    if len(candidates) != 1:
                        raise ValueError("请指定 MJCF/OBJ 主入口文件")
                    chosen = candidates[0]
                if chosen not in names:
                    raise ValueError("入口文件不在上传文件中")
                path = root / chosen
                if path.suffix.lower() == ".obj":
                    validate_obj_resources(path, root)
                    replacement = Editor()
                    loaded = trimesh.load(
                        path, force="scene", process=False, maintain_order=True
                    )
                    # Apply source scene transforms and retain per-material submeshes.
                    meshes = []
                    for node in loaded.graph.nodes_geometry:
                        transform, geom_name = loaded.graph[node]
                        mesh = loaded.geometry[geom_name].copy()
                        mesh.apply_transform(transform)
                        meshes.append(mesh)
                    if not meshes:
                        raise ValueError("OBJ 没有三角网格")
                    shared_body = None
                    for i, mesh in enumerate(meshes):
                        uv = (
                            mesh.visual.uv[mesh.faces]
                            if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None
                            else None
                        )
                        replacement.command(
                            replacement.revision,
                            "add_obj",
                            args={
                                "name": path.stem + f"_{i}",
                                "body": shared_body,
                                "scale": scale,
                                "mesh": Mesh(
                                    mesh.vertices,
                                    mesh.faces,
                                    uv,
                                    mesh.vertex_normals[mesh.faces],
                                ).json(),
                            },
                        )
                        geom = replacement.spec.geoms[-1]
                        shared_body = geom.parent.name
                        material = getattr(mesh.visual, "material", None)
                        if material is not None:
                            image = getattr(material, "image", None)
                            attrs = {
                                "rgba": (
                                    np.asarray(
                                        getattr(
                                            material, "diffuse", [180, 190, 200, 255]
                                        )
                                    )
                                    / 255
                                ).tolist()
                            }
                            if image is not None:
                                buf = io.BytesIO()
                                image.save(buf, format="PNG")
                                attrs["texture"] = base64.b64encode(
                                    buf.getvalue()
                                ).decode()
                            replacement.command(
                                replacement.revision,
                                "material",
                                "geom:" + geom.name,
                                attrs,
                            )
                else:
                    replacement = Editor(load_spec(path, root))
        if revision >= 0 and revision != editor.revision:
            raise ValueError("上传期间模型已改变，请重试导入")
        if mode == "append":
            if replacement.error:
                raise ValueError("只能将可编译模型添加到当前工程；草稿请直接打开")
            if any(n.lower().endswith(".obj") for n, _ in contents):
                ground = replacement.spec.geom("ground")
                if ground is not None:
                    replacement.spec.delete(ground)
            prefix = "imported_" + str(editor.revision) + "_"
            args = {
                "xml": replacement.spec.to_xml(),
                "assets": {
                    k: base64.b64encode(v).decode()
                    for k, v in replacement.spec.assets.items()
                },
                "prefix": prefix,
                "meta": replacement.meta,
            }
            return editor.command(editor.revision, "attach_model", "", args)
        if mode != "replace":
            raise ValueError("未知导入方式")
        replacement.revision = editor.revision + 1
        editor = replacement
        return editor.scene()

    return await asyncio.to_thread(guarded, operation)


@app.post("/api/geometry/preview")
def geometry_preview(request: dict):
    def operation():
        editor.running = False
        mesh = editor.geom_mesh(request["geom"])
        geom = next(g for g in editor.scene()["geoms"] if g["name"] == request["geom"])
        frame = {
            "pos": geom.get("editPos", geom["worldPos"]),
            "quat": geom.get("editQuat", geom["worldQuat"]),
        }
        if request.get("faces") is not None:
            selected, rest = mesh.split(request["faces"])
            if request.get("cap"):
                selected = selected.cap()
                rest = rest.cap()
            return {
                "revision": editor.revision,
                "frame": frame,
                "selected": selected.json(),
                "remaining": rest.json(),
                "diagnostics": [selected.diagnostics(), rest.diagnostics()],
            }
        if request.get("cap"):
            mesh = mesh.cap()
        result = {
            "revision": editor.revision,
            "frame": frame,
            "mesh": mesh.json(),
            "diagnostics": [mesh.diagnostics()],
        }
        if request.get("mass"):
            result["properties"] = mesh.mass_properties(
                float(request.get("density", 1000))
            )
        return result

    return guarded(operation)


@app.post("/api/simulation")
def simulation(request: dict):
    def operation():
        if (
            request["action"] in ("project_joint", "align_joints")
            and request.get("revision") != editor.revision
        ):
            raise ValueError("模型已更新，请刷新后重试")
        return editor.simulation(request["action"], request.get("values"))

    return guarded(operation)


@app.get("/api/archive")
def archive(draft: bool = True, revision: int | None = None):
    def operation():
        if revision is not None and revision != editor.revision:
            raise ValueError("页面模型与服务端不一致，已阻止导出。可能在其他页面重新导入了模型或服务已重启；请核对标定后刷新页面。")
        return editor.archive(draft)

    content = guarded(operation)
    name = "project.mjcfproj" if draft else "mjcf-model.zip"
    return Response(
        content,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/api/render")
def render():
    def operation():
        if editor.model is None:
            raise ValueError("模型尚未编译成功")
        try:
            camera = mj.MjvCamera()
            mj.mjv_defaultFreeCamera(editor.model, camera)
            with mj.Renderer(editor.model, height=480, width=640) as renderer:
                renderer.update_scene(editor.data, camera=camera)
                pixels = renderer.render()
            output = io.BytesIO()
            Image.fromarray(pixels).save(output, format="PNG")
            return Response(output.getvalue(), media_type="image/png")
        except Exception as exc:
            raise ValueError("MuJoCo 渲染不可用: " + str(exc)) from exc

    return guarded(operation)


@app.get("/api/texture/{name}")
def texture(name: str):
    def operation():
        if editor.model is None:
            raise ValueError("模型不可用")
        m = editor.model
        tid = m.texture(name).id
        width, height = int(m.tex_width[tid]), int(m.tex_height[tid])
        channels = int(m.tex_nchannel[tid])
        start = int(m.tex_adr[tid])
        pixels = m.tex_data[start : start + width * height * channels].reshape(
            height, width, channels
        )
        output = io.BytesIO()
        Image.fromarray(pixels.squeeze()).save(output, format="PNG")
        return Response(output.getvalue(), media_type="image/png")

    return guarded(operation)


@app.websocket("/api/live")
async def live(ws: WebSocket):
    global last_snapshot
    await ws.accept()
    try:
        while True:
            if lock.acquire(blocking=False):
                try:
                    last_snapshot = {
                        "revision": editor.revision,
                        "state": editor.state(),
                    }
                finally:
                    lock.release()
            await ws.send_json({**last_snapshot, "progress": progress})
            await asyncio.sleep(1 / 30)
    except (WebSocketDisconnect, RuntimeError):
        pass


async def simulation_loop():
    while True:
        if lock.acquire(blocking=False):
            try:
                if editor.running and editor.model is not None:
                    try:
                        steps = max(1, round((1 / 60) / editor.model.opt.timestep))
                        for _ in range(min(steps, 1000)):
                            mj.mj_step(editor.model, editor.data)
                        if not np.isfinite(editor.data.qpos).all() or any(
                            w.number for w in editor.data.warning
                        ):
                            editor.running = False
                            editor.error = "仿真出现数值警告，已暂停"
                    except Exception as exc:
                        editor.running = False
                        editor.error = str(exc)
            finally:
                lock.release()
        await asyncio.sleep(1 / 60)


if (ROOT / "frontend/dist").exists():
    app.mount("/", StaticFiles(directory=ROOT / "frontend/dist", html=True), name="ui")
