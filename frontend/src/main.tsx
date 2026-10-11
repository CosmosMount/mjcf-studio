import {
  FieldEditor,
  ImportForm,
  JointForm,
  ZeroForm,
  FrameForm,
  InertiaForm,
  SplitForm,
  Diagnostics,
  MaterialEditor,
  PathEditor,
} from "./forms";
import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import * as T from "three";
import {
  Box,
  Upload,
  Download,
  Undo2,
  Redo2,
  Play,
  Pause,
  RotateCcw,
  Settings2,
  Scissors,
  Move,
  Rotate3D,
  MousePointer2,
  Paintbrush,
  Scan,
  Layers,
  ChevronRight,
  Eye,
  EyeOff,
  Plus,
  Activity,
  Image as ImageIcon,
  Save,
  X,
  Maximize,
  Focus,
  Axis3D,
  Scale,
  Link2,
} from "lucide-react";
import Viewport from "./Viewport";
import Measurement from "./Measurement";
import type { Scene, State, Item, SelectionTool, V3 } from "./types";
import "./style.css";

async function api(path: string, body?: any) {
  const response = await fetch(
    "/api/" + path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers:
            body instanceof FormData
              ? {}
              : { "Content-Type": "application/json" },
          body: body instanceof FormData ? body : JSON.stringify(body),
        },
  );
  if (!response.ok) {
    let error;
    try {
      error = await response.json();
    } catch {
      error = { detail: response.statusText };
    }
    throw new Error(
      typeof error.detail === "string"
        ? error.detail
        : JSON.stringify(error.detail),
    );
  }
  return response.json();
}
function qrot(q: number[] = [1, 0, 0, 0]) {
  return new T.Quaternion(q[1], q[2], q[3], q[0]);
}
const names: Record<string, string> = {
  body: "刚体",
  geom: "几何",
  joint: "关节",
  site: "定位点",
  equality: "约束",
  tendon: "肌腱",
  actuator: "执行器",
  sensor: "传感器",
  key: "关键帧",
};
function App() {
  const [scene, setScene] = useState<Scene | null>(null),
    [state, setState] = useState<State | null>(null),
    [selected, setSelected] = useState("");
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [busy, setBusy] = useState(false),
    [tab, setTab] = useState("model");
  const [tool, setTool] = useState<SelectionTool>("object"),
    [faces, setFaces] = useState<number[]>([]),
    [selectionMode, setSelectionMode] = useState<"add" | "subtract">("add"),
    [through, setThrough] = useState(false);
  const [hidden, setHidden] = useState(new Set<string>()),
    [isolated, setIsolated] = useState<string | null>(null),
    [transparent, setTransparent] = useState(false),
    [axes, setAxes] = useState(true),
    [inertiaView, setInertiaView] = useState(false),
    [collision, setCollision] = useState(true);
  const [transform, setTransform] = useState<"none" | "translate" | "rotate">(
      "none",
    ),
    [space, setSpace] = useState<"local" | "world">("world");
  const [localPath, setLocalPath] = useState("");
  const [dialog, setDialog] = useState(""),
    [preview, setPreview] = useState<any>(null),
    [renderUrl, setRenderUrl] = useState("");
  const [search, setSearch] = useState(""),
    [inspector, setInspector] = useState("properties"),
    [fitRequest, setFitRequest] = useState(0),
    [viewDirection, setViewDirection] = useState("perspective"),
    [lengthUnit, setLengthUnit] = useState("m"),
    [angleUnit, setAngleUnit] = useState("rad");
  const [measuring, setMeasuring] = useState(false);
  const [measurePair, setMeasurePair] = useState<string[]>(["", ""]);
  const latest = useRef(scene);
  const jointRequests = useRef<{
    running: boolean;
    pending: { name: string; value: number; revision: number } | null;
  }>({ running: false, pending: null });
  latest.current = scene;
  const all = scene
    ? [
        ...scene.bodies,
        ...scene.geoms,
        ...scene.joints,
        ...scene.sites,
        ...scene.relations,
      ]
    : [];
  const item = all.find((x) => x.id === selected);
  const kind = selected.split(":")[0];
  const body =
    item &&
    (kind === "body"
      ? item
      : scene?.bodies.find((b) => b.name === item.parent));
  function accept(s: Scene) {
    if (latest.current && s.revision < latest.current.revision) return;
    latest.current = s;
    setScene(s);
    setState(s.state);
    setError("");
  }
  async function run(operation: () => Promise<any>) {
    setBusy(true);
    setError("");
    try {
      return await operation();
    } catch (e) {
      setError((e as Error).message);
      return null;
    } finally {
      setBusy(false);
    }
  }
  async function command(op: string, target = selected, args: any = {}) {
    return run(async () => {
      const s = await api("command", {
        revision: latest.current!.revision,
        op,
        target,
        args,
      });
      accept(s);
      setNotice("已应用 · " + op);
      return s;
    });
  }
  function select(id: string) {
    setSelected(id);
    setFaces([]);
    setPreview(null);
    setTransform("none");
  }
  useEffect(() => {
    api("scene")
      .then(accept)
      .catch((e) => setError(e.message));
    const ws = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/live`,
    );
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.revision === latest.current?.revision) setState(data.state);
      if (data.progress?.active) setNotice(data.progress.stage);
    };
    ws.onerror = () => setError("实时状态连接不可用，请检查本地服务");
    return () => ws.close();
  }, []);
  useEffect(() => {
    function key(e: KeyboardEvent) {
      if ((e.target as HTMLElement).matches("input,textarea,select")) return;
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        command(e.shiftKey ? "redo" : "undo", "");
      }
      if (e.key === "Escape") {
        setDialog("");
        setPreview(null);
        setFaces([]);
        setTransform("none");
      }
    }
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [scene]);
  async function example(name: string) {
    select("");
    await run(async () => {
      accept(await api("example/" + name, {}));
      setHidden(new Set());
      setIsolated(null);
    });
  }
  async function sim(action: string, values?: any) {
    if (action === "joint" && tool === "joint") {
      dragJoint(values.name, values.value);
      return;
    }
    await run(async () =>
      setState(await api("simulation", { action, values })),
    );
  }
  async function dragJoint(name: string, value: number) {
    if (!latest.current) return;
    const queue = jointRequests.current;
    queue.pending = { name, value, revision: latest.current.revision };
    if (queue.running) return;
    queue.running = true;
    try {
      while (queue.pending) {
        const request = queue.pending;
        queue.pending = null;
        if (request.revision !== latest.current?.revision) continue;
        try {
          const next: State = await api("simulation", {
            action: "project_joint",
            revision: request.revision,
            values: { name: request.name, value: request.value },
          });
          if (request.revision !== latest.current?.revision) continue;
          setState(next);
          if (next.projection?.accepted) {
            setError("");
            setNotice(
              `闭链拖动 · 残差 ${next.projection.residual.toExponential(2)}`,
            );
          } else
            setError(next.projection?.message || "闭链求解失败，保持上一姿态");
        } catch (e) {
          if (request.revision === latest.current?.revision)
            setError((e as Error).message);
        }
      }
    } finally {
      queue.running = false;
    }
  }
  async function saveLocal(path?: string) {
    await run(async () => {
      const result = await api("save", {revision: latest.current!.revision, ...(path ? {path} : {})});
      accept(await api("scene"));
      setDialog("");
      setNotice("已保存到 " + result.path);
    });
  }
  async function download(draft: boolean) {
    await run(async () => {
      const response = await fetch("/api/archive?draft=" + draft + "&revision=" + latest.current!.revision);
      if (!response.ok) throw new Error((await response.json()).detail);
      const link = document.createElement("a");
      link.href = URL.createObjectURL(await response.blob());
      link.download = draft ? "project.mjcfproj" : "mjcf-model.zip";
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
      setNotice(draft ? "工程已保存，可继续编辑草稿" : "MJCF 已验证并导出");
    });
  }
  async function transformItem(id: string, pos: V3, q: number[]) {
    const obj = all.find((i) => i.id === id);
    if (!obj || !scene) return;
    const parent = scene.bodies.find((b) => b.name === obj.parent);
    const liveParent = state?.bodies?.find((b) => b.name === obj.parent);
    const inv = qrot(liveParent?.quat || parent?.worldQuat).invert();
    const local = new T.Vector3(...pos)
      .sub(new T.Vector3(...(liveParent?.pos || parent?.worldPos || [0, 0, 0])))
      .applyQuaternion(inv);
    const localQ = inv.clone().multiply(qrot(q));
    if (id.startsWith("joint:")) {
      const axis = new T.Vector3(0, 0, 1)
        .applyQuaternion(qrot(q))
        .applyQuaternion(inv);
      await command("set", id, { pos: local.toArray(), axis: axis.toArray() });
    } else if (id.startsWith("body:")) {
      const live = state?.bodies?.find((b) => b.name === obj.name);
      const oldWorld = new T.Matrix4().compose(
        new T.Vector3(...(live?.pos || obj.worldPos!)),
        qrot(live?.quat || obj.worldQuat),
        new T.Vector3(1, 1, 1),
      );
      const newWorld = new T.Matrix4().compose(
        new T.Vector3(...pos),
        qrot(q),
        new T.Vector3(1, 1, 1),
      );
      const parentWorld = new T.Matrix4().compose(
        new T.Vector3(...(liveParent?.pos || parent?.worldPos || [0, 0, 0])),
        qrot(liveParent?.quat || parent?.worldQuat),
        new T.Vector3(1, 1, 1),
      );
      const reference = new T.Matrix4().compose(
        new T.Vector3(...(obj.fields.pos as V3)),
        qrot(obj.fields.quat),
        new T.Vector3(1, 1, 1),
      );
      const result = parentWorld
        .clone()
        .invert()
        .multiply(newWorld)
        .multiply(oldWorld.invert())
        .multiply(parentWorld)
        .multiply(reference);
      const p = new T.Vector3(),
        r = new T.Quaternion();
      result.decompose(p, r, new T.Vector3());
      await command("set", id, {
        pos: p.toArray(),
        quat: [r.w, r.x, r.y, r.z],
      });
    } else
      await command("set", id, {
        pos: local.toArray(),
        quat: [localQ.w, localQ.x, localQ.y, localQ.z],
      });
  }
  async function geometryPreview(cap = false, split = false) {
    if (!item) return;
    await run(async () => {
      const result = await api("geometry/preview", {
        geom: item.name,
        ...(split ? { faces } : {}),
        cap,
      });
      setPreview(result);
      setDialog(split ? "split" : "geometry");
    });
  }
  function connected() {
    if (!item?.geometry || !faces.length) return;
    const mesh = item.geometry;
    const adj = new Map<string, number[]>();
    mesh.faces.forEach((f, i) =>
      f.forEach((v) => {
        const key = mesh.vertices[v].map((x) => x.toPrecision(7)).join(",");
        adj.set(key, [...(adj.get(key) || []), i]);
      }),
    );
    const found = new Set(faces),
      queue = [...faces];
    while (queue.length) {
      const i = queue.pop()!;
      for (const v of mesh.faces[i])
        for (const j of adj.get(
          mesh.vertices[v].map((x) => x.toPrecision(7)).join(","),
        ) || [])
          if (!found.has(j)) {
            found.add(j);
            queue.push(j);
          }
    }
    setFaces([...found]);
  }
  const valid = state?.valid;
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <Box size={21} />
          </div>
          <strong>
            MJCF<span>STUDIO</span>
          </strong>
          <span className="local-badge">LOCAL</span>
        </div>
        <div className="project-title">
          <span className="status-dot" />
          {scene?.name || "正在连接本地服务"}
          <span className="muted">/ 工作区</span>
        </div>
        <div className="top-actions">
          <button onClick={() => setDialog("import")}>
            <Upload size={15} />
            导入
          </button>
          <button title={scene?.sourcePath || "首次保存需指定本机路径"} onClick={() => {
            if (scene?.sourcePath) void saveLocal();
            else { setLocalPath(""); setDialog("save-local"); }
          }}><Save size={15} />保存</button>
          <button onClick={() => { setLocalPath(""); setDialog("open-local"); }}>打开本地文件</button>
          <button onClick={() => download(true)}>
            <Save size={15} />
            下载工程副本
          </button>
          <button className="primary" onClick={() => download(false)}>
            <Download size={15} />
            导出 MJCF
          </button>
        </div>
      </header>
      <div className="workspace">
        <aside className="sidebar">
          <div className="sidebar-tabs">
            <button
              className={tab === "model" ? "active" : ""}
              onClick={() => setTab("model")}
            >
              <Layers size={15} />
              模型
            </button>
            <button
              className={tab === "relations" ? "active" : ""}
              onClick={() => setTab("relations")}
            >
              <Link2 size={15} />
              关联
            </button>
          </div>
          <div className="sidebar-heading">
            <span>{tab === "model" ? "场景层级" : "机构关系"}</span>
            <button title="添加关节" onClick={() => setDialog("joint")}>
              <Plus size={15} />
            </button>
          </div>
          <input
            className="search"
            placeholder="搜索模型元素…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <div className="tree">
            {tab === "model"
              ? scene?.bodies
                  .filter(
                    (b) =>
                      b.name.includes(search) ||
                      scene.geoms.some(
                        (g) => g.parent === b.name && g.name.includes(search),
                      ),
                  )
                  .map((b) => (
                    <div
                      key={b.id}
                      className="tree-body"
                      style={{
                        paddingLeft: Math.min(bodyDepth(scene!, b) * 10, 45),
                      }}
                    >
                      <button
                        className={
                          "tree-row " + (selected === b.id ? "selected" : "")
                        }
                        onClick={() => select(b.id)}
                      >
                        <ChevronRight size={12} />
                        <Box size={14} />
                        <span>
                          {b.name === "world" ? "世界坐标系" : b.name}
                        </span>
                      </button>
                      {[...scene.geoms, ...scene.joints, ...scene.sites]
                        .filter(
                          (x) => x.parent === b.name && x.name.includes(search),
                        )
                        .map((x) => (
                          <div
                            key={x.id}
                            className={
                              "tree-row child " +
                              (selected === x.id ? "selected" : "")
                            }
                          >
                            <button onClick={() => select(x.id)}>
                              {x.id.startsWith("joint") ? (
                                <Axis3D size={13} />
                              ) : x.id.startsWith("site") ? (
                                <Focus size={13} />
                              ) : (
                                <Box size={13} />
                              )}
                              <span>{x.name}</span>
                            </button>
                            {x.id.startsWith("geom") && (
                              <button
                                aria-label={"隐藏 " + x.name}
                                onClick={() =>
                                  setHidden((prev) => {
                                    const n = new Set(prev);
                                    n.has(x.id) ? n.delete(x.id) : n.add(x.id);
                                    return n;
                                  })
                                }
                              >
                                {hidden.has(x.id) ? (
                                  <EyeOff size={12} />
                                ) : (
                                  <Eye size={12} />
                                )}
                              </button>
                            )}
                          </div>
                        ))}
                    </div>
                  ))
              : scene?.relations
                  .filter((x) => x.name.includes(search))
                  .map((x) => (
                    <button
                      key={x.id}
                      className={
                        "tree-row " + (selected === x.id ? "selected" : "")
                      }
                      onClick={() => select(x.id)}
                    >
                      <Link2 size={13} />
                      <span>{x.name}</span>
                      <small>{names[x.kind!]}</small>
                    </button>
                  ))}
          </div>
          <div className="examples">
            <span className="eyebrow">示例项目</span>
            <button onClick={() => example("complex")}>
              <Activity size={15} />
              <span>
                闭环与肌腱实验台<small>约束 · 5 个自由度</small>
              </span>
              <ChevronRight size={13} />
            </button>
            <button onClick={() => example("obj")}>
              <Scissors size={15} />
              <span>
                网格拆分练习<small>OBJ · 自由选面</small>
              </span>
              <ChevronRight size={13} />
            </button>
            <button className="text-button" onClick={() => example("empty")}>
              ＋ 新建空白模型
            </button>
          </div>
          <div className="sidebar-footer">
            <span className="status-dot" />
            Python / MuJoCo <span>SI</span>
          </div>
        </aside>
        <main className="main">
          <div className="toolbar">
            <div className="tool-group">
              <Tool
                icon={<MousePointer2 />}
                active={tool === "object"}
                title="选择对象"
                action={() => {
                  setTool("object");
                  setTransform("none");
                }}
              />
              <Tool
                icon={<Scan />}
                active={tool === "face"}
                title="选择三角面"
                action={() => {
                  setTool("face");
                  setTransform("none");
                }}
              />
              <Tool
                icon={<Paintbrush />}
                active={tool === "brush"}
                title="刷选三角面"
                action={() => {
                  setTool("brush");
                  setTransform("none");
                }}
              />
              <Tool
                icon={<Maximize />}
                active={tool === "box"}
                title="框选三角面"
                action={() => {
                  setTool("box");
                  setTransform("none");
                }}
              />
            </div>
            <div className="tool-group">
              <Tool
                icon={<Link2 />}
                active={tool === "joint"}
                title="拖动关节（保持闭链）"
                action={() => {
                  setTool(tool === "joint" ? "object" : "joint");
                  setTransform("none");
                }}
              />
              <Tool
                icon={<Move />}
                active={transform === "translate"}
                title="平移"
                action={() => {
                  setTool("object");
                  setTransform(
                    transform === "translate" ? "none" : "translate",
                  );
                }}
              />
              <Tool
                icon={<Rotate3D />}
                active={transform === "rotate"}
                title="旋转"
                action={() => {
                  setTool("object");
                  setTransform(transform === "rotate" ? "none" : "rotate");
                }}
              />
              <select
                aria-label="操作坐标系"
                value={space}
                onChange={(e) => setSpace(e.target.value as any)}
              >
                <option value="world">世界</option>
                <option value="local">局部</option>
              </select>
            </div>
            <div className="tool-group">
              <Tool
                icon={<Undo2 />}
                title="撤销"
                disabled={!scene?.canUndo}
                action={() => command("undo", "")}
              />
              <Tool
                icon={<Redo2 />}
                title="重做"
                disabled={!scene?.canRedo}
                action={() => command("redo", "")}
              />
            </div>
            <div className="toolbar-right">
              <select
                aria-label="观察方向"
                value={viewDirection}
                onChange={(e) => setViewDirection(e.target.value)}
              >
                <option value="perspective">透视</option>
                <option value="front">正视</option>
                <option value="side">侧视</option>
                <option value="top">俯视</option>
              </select>
              <button
                title="聚焦选中或全部"
                onClick={() => setFitRequest((x) => x + 1)}
              >
                <Focus size={15} />
              </button>
              <button onClick={() => setDialog("view")}>
                <Settings2 size={15} />
                显示
              </button>
              <button
                onClick={() =>
                  run(async () => {
                    const response = await fetch("/api/render");
                    if (!response.ok)
                      throw new Error((await response.json()).detail);
                    if (renderUrl) URL.revokeObjectURL(renderUrl);
                    setRenderUrl(URL.createObjectURL(await response.blob()));
                    setDialog("render");
                  })
                }
              >
                <ImageIcon size={15} />
                渲染
              </button>
            </div>
          </div>
          <div className="canvas-area">
            <Viewport
              measurePair={measuring ? measurePair : []}
              viewDirection={viewDirection}
              fitRequest={fitRequest}
              scene={scene}
              state={state}
              selected={selected}
              onSelect={select}
              faces={faces}
              onFaces={setFaces}
              tool={tool}
              selectionMode={selectionMode}
              through={through}
              hidden={hidden}
              isolated={isolated}
              transparent={transparent}
              showAxes={axes}
              showInertia={inertiaView}
              showCollision={collision}
              transform={transform}
              space={space}
              onTransform={transformItem}
              onJointDrag={dragJoint}
              preview={preview?.selected || preview?.mesh}
              previewFrame={preview?.frame}
              boundaries={preview?.diagnostics?.[0]?.boundaries}
            />
            <div className="viewport-label">
              <span className="status-dot" />
              透视视图
              <span className="label-divider" />Z 向上 · 米
            </div>
            {tool !== "object" && tool !== "joint" && (
              <div className="face-toolbar">
                <strong>{faces.length} 面已选</strong>
                <select
                  aria-label="选面模式"
                  value={selectionMode}
                  onChange={(e) => setSelectionMode(e.target.value as any)}
                >
                  <option value="add">添加选择</option>
                  <option value="subtract">减去选择</option>
                </select>
                <label>
                  <input
                    type="checkbox"
                    checked={through}
                    onChange={(e) => setThrough(e.target.checked)}
                  />
                  穿透
                </label>
                <button
                  onClick={() =>
                    setFaces(
                      item?.geometry?.faces
                        .map((_, i) => i)
                        .filter((i) => !faces.includes(i)) || [],
                    )
                  }
                >
                  反选
                </button>
                <button onClick={connected}>连通扩选</button>
                <button onClick={() => setFaces([])}>清空</button>
                <button
                  className="primary"
                  disabled={!faces.length || !item?.geometry}
                  onClick={() => geometryPreview(false, true)}
                >
                  <Scissors size={14} />
                  拆分预览
                </button>
              </div>
            )}
            <div className="viewport-help">
              {tool === "joint" ? (
                "拖动关节圆点或圆环 · hinge 转动 / slide 平移 · 自动保持闭链、限位与肌腱行程（非碰撞求解）"
              ) : (
                <>
                  左键旋转视角 <span>·</span> 右键平移 <span>·</span> 滚轮缩放
                </>
              )}
            </div>
            {!scene?.bodies.some((b) => b.name !== "world") && (
              <div className="empty-guide">
                <div className="empty-icon">
                  <Box size={35} />
                </div>
                <h2>从几何到运动</h2>
                <p>导入 MJCF 或 OBJ，开始构建你的机构。</p>
                <button className="primary" onClick={() => setDialog("import")}>
                  <Upload size={15} />
                  导入模型
                </button>
                <button onClick={() => example("complex")}>打开示例</button>
              </div>
            )}
          </div>
          <div className="simulation">
            <div className="simulation-header">
              <span>
                <Activity size={15} />
                {state?.running ? "动力学仿真" : "运动学预览"}
              </span>
              <div className="sim-controls">
                <button
                  aria-label="复位"
                  disabled={!valid}
                  onClick={() => sim("reset")}
                >
                  <RotateCcw size={14} />
                </button>
                <button
                  className={state?.running ? "" : "play"}
                  aria-label={state?.running ? "暂停" : "开始仿真"}
                  disabled={!valid}
                  onClick={() => sim(state?.running ? "pause" : "play")}
                >
                  {state?.running ? <Pause size={16} /> : <Play size={16} />}
                </button>
                <span>{(state?.time || 0).toFixed(3)} s</span>
              </div>
              <span className="muted">
                闭环残差 {(state?.constraintResidual || 0).toExponential(2)}
              </span>
            </div>
            <div className="joint-sliders">
              {scene?.joints
                .filter((j) => j.type === 2 || j.type === 3)
                .map((j) => (
                  <label key={j.id}>
                    <span onClick={() => select(j.id)}>{j.name}</span>
                    <input
                      aria-label={"关节 " + j.name}
                      type="range"
                      min={
                        !j.effectiveLimited
                          ? -(j.type === 3 ? Math.PI : 1)
                          : j.fields.range[0]
                      }
                      max={
                        !j.effectiveLimited
                          ? j.type === 3
                            ? Math.PI
                            : 1
                          : j.fields.range[1]
                      }
                      step="any"
                      value={
                        state?.qpos?.[jointAddress(scene, j)] ?? j.qpos ?? 0
                      }
                      onChange={(e) =>
                        sim("joint", { name: j.name, value: +e.target.value })
                      }
                    />
                    <input
                      className="joint-value"
                      type="number"
                      step="any"
                      aria-label={"数值 " + j.name}
                      title={j.type === 3 ? angleUnit : lengthUnit}
                      value={Number(
                        (
                          (state?.qpos?.[jointAddress(scene, j)] ??
                            j.qpos ??
                            0) *
                          (j.type === 3
                            ? angleUnit === "deg"
                              ? 180 / Math.PI
                              : 1
                            : lengthUnit === "mm"
                              ? 1000
                              : 1)
                        ).toFixed(3),
                      )}
                      onChange={(e) => {
                        if (e.target.value !== "")
                          sim("joint", {
                            name: j.name,
                            value:
                              +e.target.value /
                              (j.type === 3
                                ? angleUnit === "deg"
                                  ? 180 / Math.PI
                                  : 1
                                : lengthUnit === "mm"
                                  ? 1000
                                  : 1),
                          });
                      }}
                    />
                  </label>
                ))}
              {!scene?.joints.length && (
                <p className="muted">添加关节后，可在这里预览运动。</p>
              )}
            </div>
          </div>
        </main>
        <aside className="inspector">
          <button onClick={() => setMeasuring(!measuring)}>测量与水平配合</button>
          {measuring && scene && <Measurement scene={scene} state={state} selected={selected} pair={measurePair} onPair={setMeasurePair} busy={busy}
            onAlign={async values => { await run(async () => {
              const result = await api("simulation", {action:"align_joints",revision:latest.current!.revision,values});
              setState(result); setNotice(result.projection.message);
              if (!result.projection.accepted) setError(result.projection.message);
            }); }}
            onSave={name => { if(state?.qpos) void command("save_pose", "", {name,qpos:state.qpos}); }}
            onZero={id => { select(id); setDialog("zero"); }} />}

          <div className="inspector-heading">
            <Settings2 size={15} />
            <span>属性检查器</span>
            <small>{item ? names[kind] : "未选择"}</small>
          </div>
          {item ? (
            <>
              <div className="object-title">
                <div className="object-icon">
                  {kind === "joint" ? <Axis3D size={22} /> : <Box size={22} />}
                </div>
                <div>
                  <h2>{item.name}</h2>
                  <span>{item.id}</span>
                </div>
              </div>
              <div className="inspector-tabs">
                <button
                  className={inspector === "properties" ? "active" : ""}
                  onClick={() => setInspector("properties")}
                >
                  参数
                </button>
                <button
                  className={inspector === "physical" ? "active" : ""}
                  onClick={() => setInspector("physical")}
                >
                  物理
                </button>
                <button
                  className={inspector === "appearance" ? "active" : ""}
                  onClick={() => setInspector("appearance")}
                >
                  外观
                </button>
              </div>
              <div className="inspector-content">
                {inspector === "properties" ? (
                  <>
                    <div className="section-title">
                      对象参数{" "}
                      <span>
                        {lengthUnit} / {angleUnit}
                      </span>
                    </div>
                    {!!item.references?.length && (
                      <div className="relation-links">
                        {item.references.map((ref) => (
                          <button key={ref} onClick={() => select(ref)}>
                            <Link2 size={12} />
                            {ref}
                          </button>
                        ))}
                      </div>
                    )}
                    <FieldEditor
                      key={
                        item.id + ":" + scene?.revision + lengthUnit + angleUnit
                      }
                      kind={kind}
                      jointType={item.type}
                      lengthUnit={lengthUnit}
                      angleUnit={angleUnit}
                      fields={item.fields}
                      onSave={(fields) => command("set", item.id, fields)}
                    />
                    {kind === "body" && item.name !== "world" && (
                      <div className="action-stack">
                        <button onClick={() => setDialog("reframe")}>
                          <Axis3D size={14} />
                          重定义坐标系
                        </button>
                        <button onClick={() => setDialog("reference")}>
                          <Rotate3D size={14} />
                          修改机械参考姿态
                        </button>
                        <button onClick={() => setDialog("joint")}>
                          <Plus size={14} />
                          添加关节
                        </button>
                      </div>
                    )}
                    {kind === "joint" &&
                      (item.type === 2 || item.type === 3) && (
                        <button
                          className="wide"
                          onClick={() => setDialog("zero")}
                        >
                          将指定读数设为零
                        </button>
                      )}
                    {kind === "geom" && item.geometry && (
                      <div className="action-stack">
                        <div className="stat-grid">
                          <span>
                            三角面<strong>{item.geometry.faces.length}</strong>
                          </span>
                          <span>
                            顶点<strong>{item.geometry.vertices.length}</strong>
                          </span>
                        </div>
                        <button
                          onClick={() => {
                            setTool("face");
                            setTransform("none");
                          }}
                        >
                          <Scan size={14} />
                          进入选面模式
                        </button>
                        <button onClick={() => geometryPreview(false, false)}>
                          检查网格
                        </button>
                        <button onClick={() => geometryPreview(true, false)}>
                          补洞预览
                        </button>
                      </div>
                    )}
                    {kind === "tendon" && (
                      <PathEditor
                        item={item}
                        onSave={(path) =>
                          command("replace_tendon", item.id, { path })
                        }
                      />
                    )}{" "}
                    {kind === "actuator" && (
                      <label className="form-row">
                        控制量
                        <input
                          type="number"
                          step=".05"
                          defaultValue={0}
                          onChange={(e) =>
                            sim("ctrl", {
                              name: item.name,
                              value: +e.target.value,
                            })
                          }
                        />
                      </label>
                    )}
                    {kind === "key" && (
                      <button onClick={() => sim("key", { name: item.name })}>
                        加载关键帧
                      </button>
                    )}
                  </>
                ) : inspector === "physical" ? (
                  <>
                    <div className="section-title">刚体惯性</div>
                    {body ? (
                      <>
                        <div className="stat-grid">
                          <span>
                            所属刚体<strong>{body.name}</strong>
                          </span>
                          <span>
                            质量 / kg
                            <strong>
                              {scene?.meta.stale.includes(body.name)
                                ? "待确认"
                                : body.mass?.toPrecision(5) || "—"}
                            </strong>
                          </span>
                        </div>
                        <p className="hint">
                          自动计算只使用明确选中的实体，碰撞代理不会自动重复计入。
                        </p>
                        <button
                          className="wide primary"
                          onClick={() => setDialog("inertia")}
                        >
                          <Scale size={15} />
                          计算 / 输入惯性
                        </button>
                        <button
                          className="wide"
                          onClick={() => setInertiaView(!inertiaView)}
                        >
                          {inertiaView ? "隐藏" : "显示"}质心与惯性椭球
                        </button>
                        {scene?.meta.stale.includes(body.name) && (
                          <p className="warning-text">
                            几何已改变，请重新确认惯性。
                          </p>
                        )}
                      </>
                    ) : (
                      <p className="hint">请选择刚体或几何体。</p>
                    )}
                  </>
                ) : kind === "geom" ? (
                  <MaterialEditor
                    key={item.id}
                    scene={scene!}
                    item={item}
                    onSave={(args, cap) =>
                      command(cap ? "cap_material" : "material", item.id, args)
                    }
                  />
                ) : (
                  <p className="hint">选择几何体以编辑外观材质。</p>
                )}
              </div>
              <div className="inspector-bottom">
                <button onClick={() => setIsolated(isolated ? null : item.id)}>
                  <Focus size={14} />
                  {isolated ? "取消隔离" : "隔离显示"}
                </button>
              </div>
            </>
          ) : (
            <div className="inspector-empty">
              <MousePointer2 size={24} />
              <p>选择一个模型元素</p>
              <small>检查几何、关节、惯性与材料</small>
            </div>
          )}
        </aside>
      </div>
      <footer className="statusbar">
        <span className={busy ? "warning-text" : ""}>
          <span className="status-dot" />
          {busy ? "正在计算…" : error ? "操作未完成" : notice || "就绪"}
        </span>
        <div>
          {scene?.diagnostics.length || 0} 项诊断 <span>·</span>{" "}
          {scene?.geoms.length || 0} 几何 <span>·</span>{" "}
          {scene?.joints.length || 0} 关节 <span>·</span> r
          {scene?.revision || 0}
        </div>
      </footer>
      {error || scene?.diagnostics.length ? (
        <div className="diagnostic-toast">
          {error && (
            <div className="error">
              <strong>操作失败</strong>
              <span>{error}</span>
              <button onClick={() => setError("")}>
                <X size={14} />
              </button>
            </div>
          )}
          {scene?.diagnostics.slice(0, 3).map((d, i) => (
            <div
              key={i}
              className={d.severity}
              onClick={() => d.target && select(d.target)}
            >
              <span>{d.message}</span>
            </div>
          ))}
        </div>
      ) : null}
      {dialog && (
        <div
          className={
            "modal-backdrop " +
            (["split", "geometry"].includes(dialog) ? "preview-dock" : "")
          }
        >
          <div
            className={"modal " + (dialog === "render" ? "render-modal" : "")}
          >
            <div className="modal-header">
              <h2>
                {
                  {
                    "save-local": "保存到本机文件",
                    "open-local": "打开本机原文件",
                    import: "导入模型",
                    joint: "添加关节",
                    zero: "重新标定读数零点",
                    reframe: "重定义刚体坐标系",
                    reference: "修改机械参考姿态",
                    inertia: "质量与惯性",
                    split: "网格拆分预览",
                    geometry: "网格检查与补洞",
                    view: "视图设置",
                    render: "MuJoCo 渲染预览",
                  }[dialog]
                }
              </h2>
              <button
                aria-label="关闭对话框"
                onClick={() => {
                  setDialog("");
                  setPreview(null);
                }}
              >
                <X size={18} />
              </button>
            </div>
            {(dialog === "save-local" || dialog === "open-local") && (
              <form onSubmit={e => { e.preventDefault();
                if (dialog === "save-local") void saveLocal(localPath);
                else void run(async () => {
                  accept(await api("open-local", {path:localPath, revision:latest.current!.revision}));
                  select(""); setDialog("");
                });
              }}>
                <p className="hint">填写本机 XML 或 .mjcfproj 完整路径。关联后，点击“保存”将直接写回此文件。浏览器上传无法提供原始完整路径，需要首次指定。</p>
                <label className="field vertical"><span>本地文件绝对路径</span><input aria-label="本地文件绝对路径" value={localPath} onChange={e=>setLocalPath(e.target.value)} required /></label>
                <p className="hint">保存到已有路径会覆盖该文件；XML 保存会同步写入所需资源，并合并原 include 内容。</p>
                <button className="primary wide">{dialog === "save-local" ? "保存并关联此文件" : "打开并关联原文件"}</button>
              </form>
            )}
            {dialog === "import" && (
              <ImportForm
                revision={scene?.revision || 0}
                onImport={(data) =>
                  run(async () => {
                    accept(await api("import", data));
                    select("");
                    setDialog("");
                  })
                }
              />
            )}
            {dialog === "joint" && (
              <JointForm
                scene={scene!}
                bodyName={body?.name}
                onSubmit={async (args) => {
                  if (await command("add_joint", "", args)) setDialog("");
                }}
              />
            )}
            {error && <p className="error-text" role="alert">{error}</p>}
            {dialog === "zero" && item && (
              <ZeroForm
                item={item}
                currentValue={
                  item.qposAddress === undefined
                    ? undefined
                    : state?.qpos?.[item.qposAddress]
                }
                onSubmit={async (args) => {
                  if (await command("zero", item.id, args)) setDialog("");
                }}
              />
            )}
            {(dialog === "reframe" || dialog === "reference") && item && (
              <FrameForm
                item={item}
                mode={dialog}
                onSubmit={async (args) => {
                  if (
                    await command(
                      dialog === "reframe" ? "reframe" : "pose_reference",
                      item.id,
                      args,
                    )
                  )
                    setDialog("");
                }}
              />
            )}
            {dialog === "inertia" && body && (
              <InertiaForm
                scene={scene!}
                body={body}
                onSubmit={async (args) => {
                  if (await command("inertia", body.id, args)) setDialog("");
                }}
              />
            )}
            {dialog === "split" && preview && item && (
              <SplitForm
                scene={scene!}
                item={item}
                count={faces.length}
                preview={preview}
                onPreview={(cap) => geometryPreview(cap, true)}
                onSubmit={async (args) => {
                  if (
                    await command("split", item.id, {
                      ...args,
                      faces,
                      preview_revision: preview.revision,
                    })
                  ) {
                    setDialog("");
                    setPreview(null);
                    setFaces([]);
                  }
                }}
              />
            )}
            {dialog === "geometry" && preview && (
              <>
                <Diagnostics data={preview.diagnostics} />
                <p className="hint">交互视图中的绿色线框为预览；关闭即取消。</p>
                <button
                  className="primary wide"
                  onClick={async () => {
                    if (
                      await command("cap", selected, {
                        preview_revision: preview.revision,
                      })
                    ) {
                      setDialog("");
                      setPreview(null);
                    }
                  }}
                >
                  提交补洞
                </button>
              </>
            )}
            {dialog === "view" && (
              <div className="view-options">
                <label>
                  长度单位
                  <select
                    aria-label="长度单位"
                    value={lengthUnit}
                    onChange={(e) => setLengthUnit(e.target.value)}
                  >
                    <option value="m">米 m</option>
                    <option value="mm">毫米 mm</option>
                  </select>
                </label>
                <label>
                  角度单位
                  <select
                    aria-label="角度单位"
                    value={angleUnit}
                    onChange={(e) => setAngleUnit(e.target.value)}
                  >
                    <option value="rad">弧度 rad</option>
                    <option value="deg">角度 °</option>
                  </select>
                </label>
                {[
                  ["坐标轴与关节轴", axes, setAxes],
                  ["质心与惯性椭球", inertiaView, setInertiaView],
                  ["碰撞组（group 3）", collision, setCollision],
                  ["透明显示", transparent, setTransparent],
                ].map(([label, value, setter]) => (
                  <label key={label as string}>
                    <input
                      type="checkbox"
                      checked={value as boolean}
                      onChange={(e) => (setter as any)(e.target.checked)}
                    />
                    {label as string}
                  </label>
                ))}
              </div>
            )}
            {dialog === "render" && (
              <img className="render-image" src={renderUrl} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
function bodyDepth(scene: Scene, b: Item) {
  let depth = 0;
  let current = b;
  const seen = new Set<string>();
  while (
    current.parent &&
    current.parent !== "world" &&
    !seen.has(current.parent)
  ) {
    seen.add(current.parent);
    depth++;
    const parent = scene.bodies.find((x) => x.name === current.parent);
    if (!parent) break;
    current = parent;
  }
  return depth;
}
function jointAddress(scene: Scene, j: Item) {
  if (j.qposAddress !== undefined) return j.qposAddress;
  let address = 0;
  for (const x of scene.joints) {
    if (x.id === j.id) return address;
    address += x.type === 0 ? 7 : x.type === 1 ? 4 : 1;
  }
  return address;
}
function Tool({
  icon,
  title,
  action,
  active = false,
  disabled = false,
}: {
  icon: React.ReactNode;
  title: string;
  action: () => void;
  active?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      className={"icon-button " + (active ? "active" : "")}
      title={title}
      aria-label={title}
      onClick={action}
      disabled={disabled}
    >
      {icon}
    </button>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
