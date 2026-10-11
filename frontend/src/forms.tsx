import React, { useState } from "react";
import * as T from "three";
import { Check, Upload } from "lucide-react";
import type { Scene, Item } from "./types";
function parse(text: string) {
  try {
    return JSON.parse(text);
  } catch {
    throw new Error("请输入有效 JSON 数值或数组");
  }
}
export function FieldEditor({
  fields,
  onSave,
  kind,
  jointType,
  lengthUnit,
  angleUnit,
}: {
  fields: Record<string, any>;
  onSave: (fields: any) => void;
  kind: string;
  jointType?: number;
  lengthUnit: string;
  angleUnit: string;
}) {
  const factor = (key: string) => {
    if (["pos", "ipos", "size"].includes(key))
      return lengthUnit === "mm" ? 1000 : 1;
    if (kind === "joint" && ["range", "ref", "springref"].includes(key))
      return jointType === 3
        ? angleUnit === "deg"
          ? 180 / Math.PI
          : 1
        : jointType === 2
          ? lengthUnit === "mm"
            ? 1000
            : 1
          : 1;
    return 1;
  };
  const scaled = (v: any, f: number): any =>
    Array.isArray(v)
      ? v.map((x) => scaled(x, f))
      : typeof v === "number"
        ? v * f
        : v;
  const [values, setValues] = useState<Record<string, string>>({}),
    [error, setError] = useState("");
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        try {
          const out: any = {};
          for (const [k, v] of Object.entries(values))
            out[k] =
              typeof fields[k] === "string"
                ? v
                : scaled(parse(v), 1 / factor(k));
          onSave(out);
          setError("");
        } catch (e) {
          setError((e as Error).message);
        }
      }}
    >
      {Object.entries(fields).map(([key, value]) => (
        <label className="field" key={key}>
          <span>
            {key}
            {factor(key) !== 1
              ? " · " +
                (kind === "joint" &&
                jointType === 3 &&
                ["range", "ref", "springref"].includes(key)
                  ? "°"
                  : "mm")
              : ""}
          </span>
          <input
            aria-label={key}
            value={
              values[key] ??
              (typeof value === "string"
                ? value
                : JSON.stringify(scaled(value, factor(key))))
            }
            onChange={(e) => setValues({ ...values, [key]: e.target.value })}
          />
        </label>
      ))}
      {error && <p className="error-text">{error}</p>}
      <button
        className="wide"
        type="submit"
        disabled={!Object.keys(values).length}
      >
        <Check size={14} />
        应用参数
      </button>
    </form>
  );
}
export function ImportForm({
  onImport,
  revision,
}: {
  onImport: (data: FormData) => void;
  revision: number;
}) {
  const [files, setFiles] = useState<File[]>([]),
    [entry, setEntry] = useState(""),
    [scale, setScale] = useState("1"),
    [mode, setMode] = useState("replace");
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        const data = new FormData();
        for (const f of files)
          data.append("files", f, f.webkitRelativePath || f.name);
        data.append("entry", entry);
        data.append("scale", scale);
        data.append("mode", mode);
        data.append("revision", String(revision));
        onImport(data);
      }}
    >
      <label className="field">
        <span>导入方式</span>
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="replace">打开独立工程</option>
          <option value="append">添加到当前模型</option>
        </select>
      </label>
      <div className="upload-zone">
        <Upload size={28} />
        <strong>选择模型与资源</strong>
        <p>MJCF / OBJ + MTL + 纹理，或 ZIP / 工程包</p>
        <input
          aria-label="选择模型文件"
          type="file"
          multiple
          onChange={(e) => setFiles(Array.from(e.target.files || []))}
        />
      </div>
      <label className="field">
        <span>或选择完整目录</span>
        <input
          type="file"
          {...({ webkitdirectory: "", directory: "" } as any)}
          onChange={(e) => setFiles(Array.from(e.target.files || []))}
        />
      </label>
      <p className="hint">
        已选择 {files.length} 个文件。相对路径将保留；多 XML 文件请指定主入口。
      </p>
      <label className="field">
        <span>入口文件</span>
        <select value={entry} onChange={(e) => setEntry(e.target.value)}>
          <option value="">自动识别 model.xml / 唯一模型</option>
          {files
            .filter((f) => /\.(xml|obj)$/i.test(f.name))
            .map((f) => (
              <option key={f.webkitRelativePath || f.name}>
                {f.webkitRelativePath || f.name}
              </option>
            ))}
        </select>
      </label>
      <label className="field">
        <span>OBJ 单位</span>
        <select value={scale} onChange={(e) => setScale(e.target.value)}>
          <option value="1">米（1）</option>
          <option value="0.001">毫米（0.001）</option>
          <option value="0.01">厘米（0.01）</option>
        </select>
      </label>
      <button className="primary wide" disabled={!files.length}>
        导入工作副本
      </button>
    </form>
  );
}
function JSONInput({
  label,
  value,
  set,
}: {
  label: string;
  value: string;
  set: (v: string) => void;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        aria-label={label}
        value={value}
        onChange={(e) => set(e.target.value)}
      />
    </label>
  );
}
export function JointForm({
  scene,
  bodyName,
  onSubmit,
}: {
  scene: Scene;
  bodyName?: string;
  onSubmit: (args: any) => void;
}) {
  const [body, setBody] = useState(
      bodyName && bodyName !== "world"
        ? bodyName
        : scene.bodies.find((b) => b.name !== "world")?.name || "",
    ),
    [name, setName] = useState("hinge_" + scene.joints.length),
    [type, setType] = useState("hinge"),
    [pos, setPos] = useState("[0,0,0]"),
    [axis, setAxis] = useState("[0,0,1]"),
    [range, setRange] = useState("[-3.14159,3.14159]"),
    [space, setSpace] = useState("local");
  return (
    <SafeForm
      submit={() =>
        onSubmit({
          body,
          name,
          type,
          pos: parse(pos),
          axis: parse(axis),
          range: parse(range),
          space,
        })
      }
    >
      <label className="field">
        <span>所属刚体</span>
        <select value={body} onChange={(e) => setBody(e.target.value)}>
          {scene.bodies
            .filter((b) => b.name !== "world")
            .map((b) => (
              <option key={b.name}>{b.name}</option>
            ))}
        </select>
      </label>
      <JSONInput label="关节名称" value={name} set={setName} />
      <label className="field">
        <span>关节类型</span>
        <select value={type} onChange={(e) => setType(e.target.value)}>
          <option value="hinge">转动 hinge</option>
          <option value="slide">移动 slide</option>
        </select>
      </label>
      <label className="field">
        <span>坐标空间</span>
        <select value={space} onChange={(e) => setSpace(e.target.value)}>
          <option value="local">刚体局部</option>
          <option value="world">世界参考构型</option>
        </select>
      </label>
      <JSONInput label="轴心 pos" value={pos} set={setPos} />
      <JSONInput label="轴向 axis" value={axis} set={setAxis} />
      <JSONInput label="范围 range" value={range} set={setRange} />
      <button className="primary wide">创建关节</button>
    </SafeForm>
  );
}
export function ZeroForm({
  item,
  currentValue,
  onSubmit,
}: {
  item: Item;
  currentValue?: number;
  onSubmit: (args: any) => void;
}) {
  const [offset, setOffset] = useState(String(currentValue ?? item.qpos ?? 0)),
    [resolved, setResolved] = useState("{}");
  return (
    <SafeForm
      submit={() => onSubmit({ offset: +offset, resolved: parse(resolved) })}
    >
      <p className="hint">
        标定关节：{item.name}。当前 ref：{item.fields.ref ?? 0}。
        保持几何参考姿态，采用 q新 = q旧 −
        偏移。关节限位、弹簧参考与关键帧同步换算；无法等价转换的关联项将阻止提交。
      </p>
      <JSONInput
        label="设为零的当前读数 / rad 或 m"
        value={offset}
        set={setOffset}
      />
      <label className="field vertical">
        <span>关联参数更新（可选；通常保留 {"{}"}，仅按关联错误提示填写）</span>
        <textarea
          value={resolved}
          onChange={(e) => setResolved(e.target.value)}
        />
      </label>
      <p className="hint">应用成功后本窗口关闭。导出 XML 会保存 ref，但不会把当前姿态设为默认加载姿态；如需保留姿态，请另存参考关键帧。</p>
      <button className="primary wide">应用零点标定</button>
    </SafeForm>
  );
}
export function FrameForm({
  item,
  mode,
  onSubmit,
}: {
  item: Item;
  mode: string;
  onSubmit: (args: any) => void;
}) {
  const [pos, setPos] = useState(JSON.stringify(item.fields.pos)),
    [q, setQ] = useState(JSON.stringify(item.fields.quat)),
    [space, setSpace] = useState("local");
  return (
    <SafeForm
      submit={() => onSubmit({ pos: parse(pos), quat: parse(q), space })}
    >
      <p className="hint">
        {mode === "reframe"
          ? "改变坐标表达，保持机构的世界几何、关节轴与质量分布。"
          : "改变机械参考姿态。闭环与传动可能受到影响，请检查残差及关联关系。"}
      </p>
      {mode === "reframe" && (
        <label className="field">
          <span>输入空间</span>
          <select value={space} onChange={(e) => setSpace(e.target.value)}>
            <option value="local">相对父刚体</option>
            <option value="world">世界参考构型</option>
          </select>
        </label>
      )}
      <JSONInput label="位置 [x,y,z]" value={pos} set={setPos} />
      <JSONInput label="四元数 [w,x,y,z]" value={q} set={setQ} />
      <button className="primary wide">应用修改</button>
    </SafeForm>
  );
}
export function InertiaForm({
  scene,
  body,
  onSubmit,
}: {
  scene: Scene;
  body: Item;
  onSubmit: (args: any) => void;
}) {
  const geoms = scene.geoms.filter((g) => g.parent === body.name);
  const [mode, setMode] = useState("materials"),
    [chosen, setChosen] = useState(
      geoms.filter((g) => g.fields.group !== 3).map((g) => g.name),
    ),
    [density, setDensity] = useState("1000"),
    [mass, setMass] = useState(String(body.mass || 1)),
    [center, setCenter] = useState(
      JSON.stringify(body.inertialCenter || body.fields.ipos || [0, 0, 0]),
    ),
    [tensor, setTensor] = useState(
      JSON.stringify(
        body.inertialTensor || [
          [0.01, 0, 0],
          [0, 0.01, 0],
          [0, 0, 0.01],
        ],
      ),
    );
  return (
    <SafeForm
      submit={() =>
        onSubmit({
          mode,
          geoms: chosen,
          density: +density,
          mass: +mass,
          center: parse(center),
          tensor: parse(tensor),
        })
      }
    >
      <label className="field">
        <span>惯性来源</span>
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="materials">各零件已指定材料密度</option>
          <option value="density">统一材料密度</option>
          <option value="measured">实测质量 / 均匀分布</option>
          <option value="manual">手动完整张量</option>
        </select>
      </label>
      {mode !== "manual" && (
        <>
          <div className="check-list">
            {geoms.map((g) => (
              <label key={g.name}>
                <input
                  type="checkbox"
                  checked={chosen.includes(g.name)}
                  onChange={(e) =>
                    setChosen(
                      e.target.checked
                        ? [...chosen, g.name]
                        : chosen.filter((n) => n !== g.name),
                    )
                  }
                />
                {g.name}
                <small>
                  {mode === "materials"
                    ? (scene.meta.materials[g.name]?.density || 1000) + " kg/m³"
                    : g.fields.contype
                      ? "碰撞启用"
                      : "视觉"}
                </small>
              </label>
            ))}
          </div>
          {mode === "density" && (
            <>
              <label className="field">
                <span>材料预设</span>
                <select
                  onChange={(e) => setDensity(e.target.value)}
                  value={density}
                >
                  <option value={density}>自定义</option>
                  {scene.materials.map((m) => (
                    <option key={m.name} value={m.density}>
                      {m.name} · {m.density}
                    </option>
                  ))}
                </select>
              </label>
              <JSONInput label="密度 kg/m³" value={density} set={setDensity} />
            </>
          )}
        </>
      )}
      {(mode === "manual" || mode === "measured") && (
        <JSONInput label="质量 kg" value={mass} set={setMass} />
      )}{" "}
      {mode === "manual" && (
        <>
          <JSONInput label="质心（刚体局部）" value={center} set={setCenter} />
          <label className="field vertical">
            <span>惯性张量（质心处，刚体坐标轴）</span>
            <textarea
              value={tensor}
              onChange={(e) => setTensor(e.target.value)}
            />
          </label>
        </>
      )}
      <p className="hint">
        导入的惯性只有在应用后才会被替换。开口或自交网格不能自动计算。
      </p>
      <button className="primary wide">应用刚体惯性</button>
    </SafeForm>
  );
}
export function SplitForm({
  scene,
  item,
  count,
  preview,
  onPreview,
  onSubmit,
}: {
  scene: Scene;
  item: Item;
  count: number;
  preview: any;
  onPreview: (cap: boolean) => void;
  onSubmit: (args: any) => void;
}) {
  const [name, setName] = useState("segment"),
    [newBody, setNewBody] = useState(true),
    [cap, setCap] = useState(false),
    [axis, setAxis] = useState("[0,0,1]"),
    [anchor, setAnchor] = useState("[0,0,0]"),
    [move, setMove] = useState<string[]>([]);
  return (
    <SafeForm
      submit={() =>
        onSubmit({
          name,
          new_body: newBody,
          cap,
          axis: parse(axis),
          anchor: parse(anchor),
          move,
        })
      }
    >
      <p className="hint">
        选中 {count} 个三角面。绿色预览为拆出部分，原网格在提交前不改变。
      </p>
      <Diagnostics data={preview.diagnostics} />
      <label className="check">
        <input
          type="checkbox"
          checked={cap}
          onChange={(e) => {
            setCap(e.target.checked);
            onPreview(e.target.checked);
          }}
        />
        为两侧平面边界补洞
      </label>
      <label className="check">
        <input
          type="checkbox"
          checked={newBody}
          onChange={(e) => setNewBody(e.target.checked)}
        />
        建立子刚体与转动关节
      </label>
      {newBody && (
        <>
          <JSONInput label="新刚体名称" value={name} set={setName} />
          <JSONInput
            label="关节轴心（父刚体局部）"
            value={anchor}
            set={setAnchor}
          />
          <JSONInput label="关节轴向" value={axis} set={setAxis} />
          <span className="eyebrow">随新刚体移动的子结构 / 定位点 / 几何</span>
          <div className="check-list">
            {[...scene.bodies, ...scene.sites, ...scene.geoms]
              .filter((g) => g.parent === item.parent && g.id !== item.id)
              .map((g) => (
                <label key={g.id}>
                  <input
                    type="checkbox"
                    checked={move.includes(g.id)}
                    onChange={(e) =>
                      setMove(
                        e.target.checked
                          ? [...move, g.id]
                          : move.filter((x) => x !== g.id),
                      )
                    }
                  />
                  {g.name}
                </label>
              ))}
          </div>
          <p className="hint">未勾选的元素保留在原刚体，所有引用按名称保留。</p>
        </>
      )}
      <button className="primary wide">提交拆分</button>
    </SafeForm>
  );
}
export function Diagnostics({ data }: { data: any[] }) {
  return (
    <div className="mesh-diagnostics">
      {data.map((d, i) => (
        <div key={i}>
          <strong>
            {data.length > 1 ? (i === 0 ? "选中部分" : "剩余部分") : "网格状态"}{" "}
            <span className={d.valid_solid ? "success-text" : "warning-text"}>
              {d.valid_solid ? "有效实体" : "需要检查"}
            </span>
          </strong>
          <p>
            {d.faces} 面 · {d.boundary_edges} 开口边 · {d.nonmanifold_edges}{" "}
            非流形边
          </p>
          <p>
            {d.degenerate_faces} 退化面 · {d.self_intersections?.length || 0}{" "}
            自交对（最多报告 100）
          </p>
          {d.boundary_error && (
            <p className="warning-text">{d.boundary_error}</p>
          )}
        </div>
      ))}
    </div>
  );
}
export function MaterialEditor({
  scene,
  item,
  onSave,
}: {
  scene: Scene;
  item: Item;
  onSave: (args: any, cap?: boolean) => void;
}) {
  const [preset, setPreset] = useState(0),
    [color, setColor] = useState(
      "#" +
        new T.Color(
          ...((item.rgba?.slice(0, 3) || [0.5, 0.5, 0.5]) as [
            number,
            number,
            number,
          ]),
        ).getHexString(),
    ),
    [alpha, setAlpha] = useState(item.rgba?.[3] ?? 1),
    [specular, setSpecular] = useState(item.appearance?.specular ?? 0.3),
    [shine, setShine] = useState(item.appearance?.shininess ?? 0.3),
    [reflectance, setReflectance] = useState(item.appearance?.reflectance ?? 0),
    [texture, setTexture] = useState(item.texture ? "keep" : ""),
    [repeat, setRepeat] = useState(
      JSON.stringify(item.appearance?.texrepeat || [1, 1]),
    ),
    [density, setDensity] = useState(
      String(scene.meta.materials[item.name]?.density || 2700),
    );
  return (
    <SafeForm
      submit={(intent) => {
        const c = new T.Color(color);
        onSave(
          {
            ...scene.materials[preset],
            density: +density,
            rgba: [c.r, c.g, c.b, alpha],
            specular,
            shininess: shine,
            reflectance,
            texture,
            repeat: parse(repeat),
          },
          intent === "cap",
        );
      }}
    >
      <div className="section-title">物理材料与外观</div>
      <label className="field">
        <span>材料预设</span>
        <select
          value={preset}
          onChange={(e) => {
            const i = +e.target.value;
            setPreset(i);
            setDensity(String(scene.materials[i].density));
            const c = scene.materials[i].rgba;
            setColor("#" + new T.Color(c[0], c[1], c[2]).getHexString());
          }}
        >
          {scene.materials.map((m, i) => (
            <option value={i} key={m.name}>
              {m.name}
            </option>
          ))}
        </select>
      </label>
      <JSONInput label="密度 kg/m³" value={density} set={setDensity} />
      <p className="hint">
        {scene.materials[preset].note}
        <br />
        来源：{scene.materials[preset].source}
      </p>
      <label className="field">
        <span>颜色</span>
        <input
          type="color"
          value={color}
          onChange={(e) => setColor(e.target.value)}
        />
      </label>
      {[
        ["透明度", alpha, setAlpha],
        ["高光", specular, setSpecular],
        ["光泽", shine, setShine],
        ["反射", reflectance, setReflectance],
      ].map(([name, v, set]) => (
        <label className="field" key={name as string}>
          <span>{name as string}</span>
          <input
            type="range"
            min="0"
            max="1"
            step=".01"
            value={v as number}
            onChange={(e) => (set as any)(+e.target.value)}
          />
        </label>
      ))}
      <label className="field">
        <span>程序纹理</span>
        <select
          value={
            texture === "keep"
              ? "keep"
              : texture === "checker"
                ? "checker"
                : texture
                  ? "custom"
                  : ""
          }
          onChange={(e) => setTexture(e.target.value)}
        >
          {item.texture && <option value="keep">保留原纹理</option>}
          <option value="">纯色</option>
          <option value="checker">棋盘纹理</option>
          {texture && texture !== "checker" && (
            <option value="custom">本地图片</option>
          )}
        </select>
      </label>
      <label className="field vertical">
        <span>导入本地纹理</span>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            const reader = new FileReader();
            reader.onload = () =>
              setTexture((reader.result as string).split(",")[1]);
            reader.readAsDataURL(f);
          }}
        />
      </label>
      <JSONInput label="纹理平铺 [u,v]" value={repeat} set={setRepeat} />
      <button className="primary wide">应用材料</button>
      {scene.meta.meshes[item.mesh || ""]?.materials?.some(
        (v: number) => v > 0,
      ) && (
        <button type="submit" value="cap" className="wide">
          仅应用到补面外观
        </button>
      )}
      <p className="hint">修改外观不会自动改变摩擦或已有惯性。</p>
    </SafeForm>
  );
}
export function PathEditor({
  item,
  onSave,
}: {
  item: Item;
  onSave: (path: any) => void;
}) {
  const map: Record<number, string> = {
    1: "joint",
    2: "pulley",
    3: "site",
    4: "geom",
    5: "geom",
  };
  const [text, setText] = useState(
    JSON.stringify(
      item.path?.map((p) => ({ ...p, type: map[p.type] })),
      null,
      2,
    ),
  );
  return (
    <SafeForm submit={() => onSave(parse(text))}>
      <div className="section-title">肌腱路径</div>
      <textarea
        aria-label="肌腱路径"
        className="code-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button className="wide">应用路径</button>
    </SafeForm>
  );
}
function SafeForm({
  submit,
  children,
}: {
  submit: (intent?: string) => void;
  children: React.ReactNode;
}) {
  const [error, setError] = useState("");
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        try {
          submit(
            (e.nativeEvent as SubmitEvent).submitter instanceof
              HTMLButtonElement
              ? ((e.nativeEvent as SubmitEvent).submitter as HTMLButtonElement)
                  .value
              : undefined,
          );
          setError("");
        } catch (e) {
          setError((e as Error).message);
        }
      }}
    >
      {children}
      {error && <p className="error-text">{error}</p>}
    </form>
  );
}
