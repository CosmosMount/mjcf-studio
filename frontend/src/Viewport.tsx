import { useEffect, useRef } from "react";
import * as T from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { Scene, State, Item, MeshData, SelectionTool, V3 } from "./types";

function quat(q?: number[]) {
  return q ? new T.Quaternion(q[1], q[2], q[3], q[0]) : new T.Quaternion();
}
function geometry(data: MeshData) {
  const g = new T.BufferGeometry();
  const pos = data.faces.flatMap((f) => f.flatMap((v) => data.vertices[v]));
  g.setAttribute("position", new T.Float32BufferAttribute(pos, 3));
  if (data.uv)
    g.setAttribute("uv", new T.Float32BufferAttribute(data.uv.flat(2), 2));
  if (data.normals)
    g.setAttribute(
      "normal",
      new T.Float32BufferAttribute(data.normals.flat(2), 3),
    );
  else g.computeVertexNormals();
  return g;
}
function primitive(item: Item) {
  const s = item.size || item.fields.size || [0.1, 0.1, 0.1];
  switch (item.type) {
    case 0: {
      const g = new T.PlaneGeometry(s[0] * 2 || 10, s[1] * 2 || 10);
      return g;
    }
    case 2:
      return new T.SphereGeometry(s[0], 24, 16);
    case 3: {
      const g = new T.CapsuleGeometry(s[0], s[1] * 2, 8, 24);
      g.rotateX(Math.PI / 2);
      return g;
    }
    case 4: {
      const g = new T.SphereGeometry(1, 24, 16);
      g.scale(s[0], s[1], s[2]);
      return g;
    }
    case 5: {
      const g = new T.CylinderGeometry(s[0], s[0], s[1] * 2, 32);
      g.rotateX(Math.PI / 2);
      return g;
    }
    case 6:
      return new T.BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2);
    default:
      return item.geometry
        ? geometry(item.geometry)
        : new T.BoxGeometry(0.02, 0.02, 0.02);
  }
}

type Props = {
  measurePair?: string[];
  viewDirection: string;
  fitRequest: number;
  scene: Scene | null;
  state: State | null;
  selected: string;
  onSelect: (id: string) => void;
  faces: number[];
  onFaces: (faces: number[]) => void;
  tool: SelectionTool;
  selectionMode: "add" | "subtract";
  through: boolean;
  hidden: Set<string>;
  isolated: string | null;
  transparent: boolean;
  showAxes: boolean;
  showInertia: boolean;
  showCollision: boolean;
  transform: "none" | "translate" | "rotate";
  space: "local" | "world";
  onTransform: (id: string, pos: V3, q: number[]) => void;
  onJointDrag: (name: string, value: number) => void;
  preview?: MeshData | null;
  previewFrame?: { pos: V3; quat: number[] };
  boundaries?: V3[][];
};
export default function Viewport(props: Props) {
  const host = useRef<HTMLDivElement>(null);
  const current = useRef(props);
  current.current = props;
  const engine = useRef<any>(null);
  useEffect(() => {
    const e = engine.current;
    const pair = props.measurePair || [];
    const positions = pair.map(n => props.state?.joints?.find(j => j.name === n)?.pos);
    if (!e || positions.length !== 2 || positions.some(p => !p)) return;
    const geometry = new T.BufferGeometry().setFromPoints(positions.map(p => new T.Vector3(...p!)));
    const material = new T.LineBasicMaterial({color: 0x67ffcd, depthTest: false});
    const line = new T.Line(geometry, material); line.renderOrder = 100;
    e.scene.add(line);
    return () => { e.scene.remove(line); geometry.dispose(); material.dispose(); };
  }, [props.measurePair, props.state]);
  useEffect(() => {
    const element = host.current!;
    const scene = new T.Scene();
    scene.background = new T.Color("#101923");
    const camera = new T.PerspectiveCamera(42, 1, 0.005, 1000);
    camera.up.set(0, 0, 1);
    camera.position.set(2.1, -3, 1.9);
    const renderer = new T.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = T.SRGBColorSpace;
    element.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.3);
    controls.enableDamping = true;
    scene.add(new T.HemisphereLight(0xe6f0ff, 0x394552, 2.3));
    const light = new T.DirectionalLight(0xffffff, 3);
    light.position.set(3, -4, 6);
    scene.add(light);
    const grid = new T.GridHelper(8, 40, 0x3a5264, 0x243444);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = -0.002;
    scene.add(grid);
    const root = new T.Group();
    scene.add(root);
    const overlays = new T.Group();
    scene.add(overlays);
    const dynamic = new T.Group();
    scene.add(dynamic);
    const jointHandles = new T.Group();
    scene.add(jointHandles);
    const gizmo = new TransformControls(camera, renderer.domElement);
    gizmo.setSize(0.75);
    scene.add(gizmo.getHelper());
    const anchor = new T.Object3D();
    scene.add(anchor);
    gizmo.addEventListener("dragging-changed", (event: any) => {
      controls.enabled = !event.value;
      if (!event.value) {
        const p = current.current;
        p.onTransform(p.selected, anchor.position.toArray() as V3, [
          anchor.quaternion.w,
          anchor.quaternion.x,
          anchor.quaternion.y,
          anchor.quaternion.z,
        ]);
      }
    });
    const resize = new ResizeObserver(() => {
      const w = element.clientWidth,
        h = element.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    resize.observe(element);
    const ray = new T.Raycaster();
    const pointer = new T.Vector2();
    let down: { x: number; y: number } | null = null;
    let moved = false;
    let jointDrag: {
      name: string;
      type: number;
      value: number;
      x: number;
      y: number;
      plane: T.Plane;
      axis: T.Vector3;
      center: T.Vector3;
      previous: T.Vector3 | null;
      angle: number;
      screenAxis: T.Vector2;
      axisStep: number;
      perPixel: number;
    } | null = null;
    function updatePointer(x: number, y: number) {
      camera.updateMatrixWorld(true);
      root.updateMatrixWorld(true);
      const r = renderer.domElement.getBoundingClientRect();
      pointer.set(
        ((x - r.left) / r.width) * 2 - 1,
        (-(y - r.top) / r.height) * 2 + 1,
      );
      ray.setFromCamera(pointer, camera);
    }
    function selectableMeshes() {
      return root.children.filter(
        (o) => o instanceof T.Mesh && o.visible && o.userData.item?.type !== 0,
      ) as T.Mesh[];
    }
    function applyFaces(ids: number[]) {
      const p = current.current;
      const next = new Set(p.faces);
      for (const id of ids) {
        if (p.selectionMode === "subtract") next.delete(id);
        else next.add(id);
      }
      p.onFaces([...next]);
    }
    function screenFaces(
      x: number,
      y: number,
      start?: { x: number; y: number },
    ) {
      camera.updateMatrixWorld(true);
      root.updateMatrixWorld(true);
      const p = current.current;
      const mesh = root.children.find((o) => o.userData.id === p.selected) as
        T.Mesh | undefined;
      if (!mesh?.geometry) return;
      const r = renderer.domElement.getBoundingClientRect();
      const a = mesh.geometry.getAttribute("position");
      const hits: number[] = [];
      const point = new T.Vector3();
      for (let i = 0; i < a.count; i += 3) {
        point.set(0, 0, 0);
        for (let j = 0; j < 3; j++)
          point.add(new T.Vector3().fromBufferAttribute(a, i + j));
        point.multiplyScalar(1 / 3).applyMatrix4(mesh.matrixWorld);
        const projected = point.clone().project(camera);
        if (projected.z < -1 || projected.z > 1) continue;
        const sx = ((projected.x + 1) * r.width) / 2 + r.left,
          sy = ((-projected.y + 1) * r.height) / 2 + r.top;
        const inside = start
          ? sx >= Math.min(start.x, x) &&
            sx <= Math.max(start.x, x) &&
            sy >= Math.min(start.y, y) &&
            sy <= Math.max(start.y, y)
          : Math.hypot(sx - x, sy - y) < 28;
        if (!inside) continue;
        if (!p.through) {
          updatePointer(sx, sy);
          const hit = ray.intersectObjects(selectableMeshes(), false)[0];
          if (hit?.object !== mesh || hit.faceIndex !== i / 3) continue;
        }
        hits.push(i / 3);
      }
      applyFaces(hits);
    }
    const box = document.createElement("div");
    box.className = "selection-box";
    element.appendChild(box);
    function onDown(e: PointerEvent) {
      if (e.button !== 0 || gizmo.dragging) return;
      const p = current.current;
      if (p.tool === "joint") {
        updatePointer(e.clientX, e.clientY);
        jointHandles.updateMatrixWorld(true);
        const hits = ray.intersectObjects(jointHandles.children, true);
        const hit =
          hits.find((h) => "joint:" + h.object.userData.joint === p.selected) ||
          hits[0];
        if (hit) {
          const item = p.scene!.joints.find(
            (j) => j.name === hit.object.userData.joint,
          )!;
          const live = p.state?.joints?.find((j) => j.name === item.name);
          const center = new T.Vector3(...(live?.pos || item.worldPos!));
          const axis = new T.Vector3(
            ...(live?.axis || item.worldAxis!),
          ).normalize();
          const plane = new T.Plane().setFromNormalAndCoplanarPoint(
            axis,
            center,
          );
          const perPixel =
            (2 *
              camera.position.distanceTo(center) *
              Math.tan(T.MathUtils.degToRad(camera.fov / 2))) /
            element.clientHeight;
          const intersection = ray.ray.intersectPlane(plane, new T.Vector3());
          const previous =
            intersection &&
            Math.abs(ray.ray.direction.dot(axis)) > 0.15 &&
            intersection.distanceTo(center) > perPixel * 4
              ? intersection.sub(center).normalize()
              : null;
          const start = center.clone().project(camera);
          const axisStep = camera.position.distanceTo(center) * 0.1;
          const end = center
            .clone()
            .addScaledVector(axis, axisStep)
            .project(camera);
          jointDrag = {
            name: item.name,
            type: item.type!,
            value: p.state?.qpos?.[item.qposAddress!] ?? item.qpos ?? 0,
            x: e.clientX,
            y: e.clientY,
            plane,
            axis,
            center,
            previous,
            angle: 0,
            perPixel,
            axisStep,
            screenAxis: new T.Vector2(
              ((end.x - start.x) * element.clientWidth) / 2,
              (-(end.y - start.y) * element.clientHeight) / 2,
            ),
          };
          p.onSelect(item.id);
          controls.enabled = false;
          down = null;
          renderer.domElement.setPointerCapture(e.pointerId);
          e.stopImmediatePropagation();
          e.preventDefault();
          return;
        }
      }
      down = { x: e.clientX, y: e.clientY };
      moved = false;
      if (current.current.tool === "brush" || current.current.tool === "box") {
        controls.enabled = false;
        renderer.domElement.setPointerCapture(e.pointerId);
        if (current.current.tool === "brush") screenFaces(e.clientX, e.clientY);
      }
    }
    function onMove(e: PointerEvent) {
      if (jointDrag) {
        const drag = jointDrag;
        const dx = e.clientX - drag.x,
          dy = e.clientY - drag.y;
        let delta: number;
        if (drag.type === 3) {
          updatePointer(e.clientX, e.clientY);
          const point = ray.ray.intersectPlane(drag.plane, new T.Vector3());
          if (
            drag.previous &&
            point &&
            point.distanceTo(drag.center) > drag.perPixel * 2
          ) {
            const next = point.sub(drag.center).normalize();
            drag.angle += Math.atan2(
              drag.axis.dot(new T.Vector3().crossVectors(drag.previous, next)),
              drag.previous.dot(next),
            );
            drag.previous = next;
            delta = drag.angle;
          } else delta = drag.previous ? drag.angle : (dx - dy) * 0.01;
        } else {
          delta =
            drag.screenAxis.lengthSq() > 4
              ? (new T.Vector2(dx, dy).dot(drag.screenAxis) * drag.axisStep) /
                drag.screenAxis.lengthSq()
              : -dy * drag.perPixel;
        }
        current.current.onJointDrag(drag.name, drag.value + delta);
        e.stopImmediatePropagation();
        return;
      }
      if (!down || gizmo.dragging) return;
      moved = moved || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4;
      if (current.current.tool === "brush") screenFaces(e.clientX, e.clientY);
      if (current.current.tool === "box") {
        const r = element.getBoundingClientRect();
        Object.assign(box.style, {
          display: "block",
          left: Math.min(down.x, e.clientX) - r.left + "px",
          top: Math.min(down.y, e.clientY) - r.top + "px",
          width: Math.abs(e.clientX - down.x) + "px",
          height: Math.abs(e.clientY - down.y) + "px",
        });
      }
    }
    function onUp(e: PointerEvent) {
      if (jointDrag) {
        jointDrag = null;
        controls.enabled = true;
        if (renderer.domElement.hasPointerCapture(e.pointerId))
          renderer.domElement.releasePointerCapture(e.pointerId);
        e.stopImmediatePropagation();
        return;
      }
      box.style.display = "none";
      controls.enabled = true;
      if (!down || gizmo.dragging) {
        down = null;
        return;
      }
      const p = current.current;
      if (p.tool === "box") {
        screenFaces(e.clientX, e.clientY, down);
        down = null;
        return;
      }
      if (!moved) {
        updatePointer(e.clientX, e.clientY);
        const hits = ray.intersectObjects(selectableMeshes(), false);
        const hit =
          p.tool === "face" && p.through
            ? hits.find((h) => h.object.userData.id === p.selected) || hits[0]
            : hits[0];
        if (hit) {
          const id = hit.object.userData.id;
          if (p.tool === "face" && p.selected === id) {
            applyFaces(
              (p.through ? hits.filter((h) => h.object === hit.object) : [hit])
                .map((h) => h.faceIndex!)
                .filter((n) => n != null),
            );
          } else p.onSelect(id);
        }
      }
      down = null;
    }
    renderer.domElement.addEventListener("pointerdown", onDown, true);
    renderer.domElement.addEventListener("pointermove", onMove, true);
    renderer.domElement.addEventListener("pointerup", onUp, true);
    renderer.domElement.addEventListener("pointercancel", onUp, true);
    let frame = 0;
    function tick() {
      controls.update();
      for (const handle of jointHandles.children) {
        const pixel =
          (2 *
            camera.position.distanceTo(handle.position) *
            Math.tan(T.MathUtils.degToRad(camera.fov / 2))) /
          Math.max(element.clientHeight, 1);
        handle.scale.setScalar(pixel * (handle.userData.ring ? 38 : 8));
      }
      renderer.render(scene, camera);
      frame = requestAnimationFrame(tick);
    }
    tick();
    engine.current = {
      scene,
      root,
      overlays,
      dynamic,
      jointHandles,
      camera,
      renderer,
      controls,
      gizmo,
      anchor,
      items: new Map(),
    };
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      gizmo.dispose();
      controls.dispose();
      scene.traverse((o: any) => {
        o.geometry?.dispose();
        if (o.material) {
          (Array.isArray(o.material) ? o.material : [o.material]).forEach(
            (m: any) => m.dispose(),
          );
        }
      });
      renderer.dispose();
      element.replaceChildren();
      engine.current = null;
    };
  }, []);
  useEffect(() => {
    const e = engine.current;
    if (!e) return;
    disposeGroup(e.jointHandles);
    if (props.tool !== "joint" || !props.scene) return;
    for (const joint of props.scene.joints) {
      if (joint.type !== 2 && joint.type !== 3) continue;
      const selected = joint.id === props.selected;
      const live = props.state?.joints?.find((j) => j.name === joint.name);
      const add = (ring: boolean) => {
        const mesh = new T.Mesh(
          ring
            ? new T.TorusGeometry(1, 0.065, 8, 64)
            : new T.SphereGeometry(1, 12, 8),
          new T.MeshBasicMaterial({
            color: selected ? 0xffce7a : 0x6edbc5,
            depthTest: false,
          }),
        );
        mesh.position.fromArray(live?.pos || joint.worldPos!);
        mesh.quaternion.setFromUnitVectors(
          new T.Vector3(0, 0, 1),
          new T.Vector3(...(live?.axis || joint.worldAxis!)).normalize(),
        );
        mesh.userData = { joint: joint.name, ring };
        mesh.renderOrder = 20;
        const pixel =
          (2 *
            e.camera.position.distanceTo(mesh.position) *
            Math.tan(T.MathUtils.degToRad(e.camera.fov / 2))) /
          Math.max(host.current!.clientHeight, 1);
        mesh.scale.setScalar(pixel * (ring ? 38 : 8));
        e.jointHandles.add(mesh);
      };
      add(false);
      if (selected && joint.type === 3) add(true);
    }
  }, [props.scene, props.tool, props.selected]);
  useEffect(() => {
    const e = engine.current;
    if (!e || !props.scene) return;
    disposeGroup(e.root);
    disposeGroup(e.overlays);
    e.items.clear();
    for (const item of props.scene.geoms) {
      const color = item.rgba || item.fields.rgba || [0.7, 0.7, 0.7, 1];
      const material = new T.MeshStandardMaterial({
        color: new T.Color(color[0], color[1], color[2]),
        opacity: color[3],
        transparent: color[3] < 1,
        roughness: 1 - (item.appearance?.shininess || 0.2) * 0.7,
        metalness: item.appearance?.reflectance || 0,
        side: T.DoubleSide,
      });
      if (item.texture) {
        const texture = new T.TextureLoader().load(
          "/api/texture/" +
            encodeURIComponent(item.texture) +
            "?v=" +
            props.scene.revision,
        );
        texture.colorSpace = T.SRGBColorSpace;
        texture.wrapS = texture.wrapT = T.RepeatWrapping;
        texture.repeat.fromArray(item.appearance?.texrepeat || [1, 1]);
        material.map = texture;
      }
      const mesh = new T.Mesh(primitive(item), material);
      mesh.position.fromArray(item.worldPos || item.fields.pos || [0, 0, 0]);
      mesh.quaternion.copy(quat(item.worldQuat || item.fields.quat));
      mesh.userData = { id: item.id, item };
      e.root.add(mesh);
      e.items.set(item.name, mesh);
    }
    if (props.showAxes) {
      for (const b of props.scene.bodies) {
        if (b.name === "world") continue;
        const axis = new T.AxesHelper(0.1);
        axis.position.fromArray(b.worldPos || [0, 0, 0]);
        axis.quaternion.copy(quat(b.worldQuat));
        axis.userData = { kind: "body", name: b.name };
        e.overlays.add(axis);
      }
      for (const j of props.scene.joints) {
        if (j.worldPos && j.worldAxis) {
          const arrow = new T.ArrowHelper(
            new T.Vector3(...j.worldAxis),
            new T.Vector3(...j.worldPos),
            0.18,
            0xf5bb62,
            0.035,
            0.015,
          );
          arrow.userData = { kind: "joint", name: j.name };
          e.overlays.add(arrow);
        }
      }
      for (const s of props.scene.sites) {
        if (!s.worldPos) continue;
        const marker = new T.Mesh(
          new T.SphereGeometry(0.008),
          new T.MeshBasicMaterial({ color: 0xe9bd77 }),
        );
        marker.position.fromArray(s.worldPos);
        marker.userData = { kind: "site", name: s.name };
        e.overlays.add(marker);
      }
    }
    if (props.showInertia)
      for (const b of props.scene.bodies) {
        if (!b.com || !b.inertia || !b.mass) continue;
        const dot = new T.Mesh(
          new T.SphereGeometry(0.009),
          new T.MeshBasicMaterial({ color: 0xffb870 }),
        );
        dot.position.fromArray(b.com);
        dot.userData = { kind: "com", name: b.name };
        e.overlays.add(dot);
        const [a, c, d] = b.inertia;
        const radii = [c + d - a, a + d - c, a + c - d].map((x) =>
          Math.sqrt(Math.max(0, (5 * x) / (2 * b.mass!))),
        );
        const ellipsoid = new T.Mesh(
          new T.SphereGeometry(1, 20, 12),
          new T.MeshBasicMaterial({
            color: 0xffc572,
            wireframe: true,
            transparent: true,
            opacity: 0.35,
          }),
        );
        ellipsoid.scale.fromArray(radii);
        ellipsoid.position.fromArray(b.com);
        ellipsoid.quaternion.copy(quat(b.inertialQuat));
        ellipsoid.userData = { kind: "inertia", name: b.name };
        e.overlays.add(ellipsoid);
      }
  }, [props.scene, props.showAxes, props.showInertia]);
  useEffect(() => {
    const e = engine.current;
    if (!e || !props.scene) return;
    const bodySelected = props.selected.startsWith("body:")
      ? props.selected.slice(5)
      : "";
    const belongs = (parent: string, id: string) => {
      const ancestors = new Set<string>();
      let name = parent;
      while (name && !ancestors.has(name)) {
        if ("body:" + name === id) return true;
        ancestors.add(name);
        name = props.scene!.bodies.find((b) => b.name === name)?.parent || "";
      }
      return false;
    };
    for (const mesh of e.root.children as T.Mesh[]) {
      const item = mesh.userData.item as Item;
      mesh.visible =
        !props.hidden.has(item.id) &&
        (!props.isolated ||
          props.isolated === item.id ||
          belongs(item.parent!, props.isolated)) &&
        (props.showCollision || !(item.fields.group === 3));
      const selected =
        item.id === props.selected ||
        (bodySelected && belongs(item.parent!, "body:" + bodySelected));
      const mat = mesh.material as T.MeshStandardMaterial;
      mat.emissive.set(selected ? 0x143d44 : 0);
      mat.opacity = props.transparent
        ? selected
          ? 0.7
          : 0.2
        : (item.rgba?.[3] ?? 1);
      mat.transparent = mat.opacity < 1;
      const old = mesh.getObjectByName("selected-faces");
      if (old) {
        mesh.remove(old);
        (old as T.Mesh).geometry.dispose();
      }
      if (item.id === props.selected && props.faces.length && item.geometry) {
        const faces = props.faces
          .filter((i) => i < item.geometry!.faces.length)
          .map((i) => item.geometry!.faces[i]);
        const overlay = new T.Mesh(
          geometry({ ...item.geometry, faces, uv: null, normals: null }),
          new T.MeshBasicMaterial({
            color: 0x58dec3,
            transparent: true,
            opacity: 0.55,
            side: T.DoubleSide,
            depthWrite: false,
            polygonOffset: true,
            polygonOffsetFactor: -2,
          }),
        );
        overlay.name = "selected-faces";
        mesh.add(overlay);
      }
    }
    e.gizmo.detach();
    const item = [
      ...props.scene.bodies,
      ...props.scene.geoms,
      ...props.scene.joints,
      ...props.scene.sites,
    ].find((i) => i.id === props.selected);
    if (item?.worldPos && props.transform !== "none") {
      e.anchor.position.fromArray(item.editPos || item.worldPos);
      if (item.worldAxis)
        e.anchor.quaternion.setFromUnitVectors(
          new T.Vector3(0, 0, 1),
          new T.Vector3(...item.worldAxis),
        );
      else e.anchor.quaternion.copy(quat(item.editQuat || item.worldQuat));
      e.gizmo.setMode(props.transform);
      e.gizmo.setSpace(props.space);
      e.gizmo.attach(e.anchor);
    }
  }, [
    props.scene,
    props.selected,
    props.faces,
    props.hidden,
    props.isolated,
    props.transparent,
    props.showCollision,
    props.transform,
    props.space,
  ]);
  useEffect(() => {
    const e = engine.current;
    if (!e || !props.state) return;
    for (const handle of e.jointHandles.children) {
      const live = props.state.joints?.find(
        (j) => j.name === handle.userData.joint,
      );
      if (live) {
        handle.position.fromArray(live.pos);
        handle.quaternion.setFromUnitVectors(
          new T.Vector3(0, 0, 1),
          new T.Vector3(...live.axis).normalize(),
        );
      }
    }
    for (const item of props.state.geoms || []) {
      const mesh = e.items.get(item.name);
      if (mesh) {
        mesh.position.fromArray(item.pos);
        mesh.quaternion.copy(quat(item.quat));
      }
    }
    disposeGroup(e.dynamic);
    for (const o of e.overlays.children) {
      const { kind, name } = o.userData;
      const b = props.state.bodies?.find((v) => v.name === name);
      const j = props.state.joints?.find((v) => v.name === name);
      const site = props.state.sites?.find((v) => v.name === name);
      if (kind === "body" && b) {
        o.position.fromArray(b.pos);
        o.quaternion.copy(quat(b.quat));
      }
      if (kind === "joint" && j) {
        o.position.fromArray(j.pos);
        o.setDirection(new T.Vector3(...j.axis));
      }
      if (kind === "site" && site) {
        o.position.fromArray(site.pos);
        o.quaternion.copy(quat(site.quat));
      }
      if ((kind === "com" || kind === "inertia") && b) {
        o.position.fromArray(b.com);
        if (kind === "inertia") o.quaternion.copy(quat(b.inertialQuat));
      }
    }
    if (!e.gizmo.dragging && props.transform !== "none") {
      const [kind, name] = [
        props.selected.slice(0, props.selected.indexOf(":")),
        props.selected.slice(props.selected.indexOf(":") + 1),
      ];
      const live =
        kind === "body"
          ? props.state.bodies?.find((v) => v.name === name)
          : kind === "geom"
            ? props.state.geoms?.find((v) => v.name === name)
            : kind === "joint"
              ? props.state.joints?.find((v) => v.name === name)
              : props.state.sites?.find((v) => v.name === name);
      if (live) {
        e.anchor.position.fromArray(
          "editPos" in live && live.editPos ? live.editPos : live.pos,
        );
        if ("axis" in live)
          e.anchor.quaternion.setFromUnitVectors(
            new T.Vector3(0, 0, 1),
            new T.Vector3(...live.axis),
          );
        else
          e.anchor.quaternion.copy(
            quat(
              "editQuat" in live && live.editQuat
                ? (live.editQuat as number[])
                : live.quat,
            ),
          );
      }
    }
    const selectedRelation = props.scene?.relations.find(
      (r) => r.id === props.selected,
    );
    if (selectedRelation?.references) {
      const positions: T.Vector3[] = [];
      const visit = (id: string, depth = 0) => {
        if (depth > 2) return;
        const [kind, name] = [
          id.slice(0, id.indexOf(":")),
          id.slice(id.indexOf(":") + 1),
        ];
        const list =
          kind === "body"
            ? props.state?.bodies
            : kind === "joint"
              ? props.state?.joints
              : kind === "site"
                ? props.state?.sites
                : kind === "geom"
                  ? props.state?.geoms
                  : [];
        const found = list?.find((v) => v.name === name);
        if (found) {
          positions.push(new T.Vector3(...found.pos));
        } else
          props.scene?.relations
            .find((r) => r.id === id)
            ?.references?.forEach((r) => visit(r, depth + 1));
      };
      selectedRelation.references.forEach((id) => visit(id));
      if (positions.length > 1)
        e.dynamic.add(
          new T.Line(
            new T.BufferGeometry().setFromPoints(positions),
            new T.LineBasicMaterial({
              color: 0xffce7a,
              depthTest: false,
              transparent: true,
              opacity: 0.9,
            }),
          ),
        );
      for (const pos of positions) {
        const dot = new T.Mesh(
          new T.SphereGeometry(0.014),
          new T.MeshBasicMaterial({ color: 0xffce7a, depthTest: false }),
        );
        dot.position.copy(pos);
        e.dynamic.add(dot);
      }
    }
    for (const p of props.state.contacts || []) {
      const dot = new T.Mesh(
        new T.SphereGeometry(0.006),
        new T.MeshBasicMaterial({ color: 0xff725c }),
      );
      dot.position.fromArray(p);
      e.dynamic.add(dot);
    }
    for (const line of props.state.tendons || []) {
      if (line.length < 2) continue;
      e.dynamic.add(
        new T.Line(
          new T.BufferGeometry().setFromPoints(
            line.map((p) => new T.Vector3(...p)),
          ),
          new T.LineBasicMaterial({ color: 0xc78ef2 }),
        ),
      );
    }
  }, [props.state, props.selected]);
  useEffect(() => {
    const e = engine.current;
    if (!e) return;
    const old = e.scene.getObjectByName("preview");
    if (old) {
      e.scene.remove(old);
      disposeGroup(old as T.Group);
    }
    if (!props.preview) return;
    const item = props.scene?.geoms.find((i) => i.id === props.selected);
    if (!item) return;
    const group = new T.Group();
    group.name = "preview";
    const mesh = new T.Mesh(
      geometry(props.preview),
      new T.MeshStandardMaterial({
        color: 0x5ce0c1,
        transparent: true,
        opacity: 0.6,
        wireframe: true,
      }),
    );
    // Preview mesh coordinates are in the original geom frame, not the compiler-centered frame.
    group.position.fromArray(
      props.previewFrame?.pos || item.editPos || item.worldPos!,
    );
    group.quaternion.copy(
      quat(props.previewFrame?.quat || item.editQuat || item.worldQuat),
    );
    group.add(mesh);
    for (const loop of props.boundaries || []) {
      const points = [...loop, loop[0]].map((p) => new T.Vector3(...p));
      group.add(
        new T.Line(
          new T.BufferGeometry().setFromPoints(points),
          new T.LineBasicMaterial({ color: 0xff987b, depthTest: false }),
        ),
      );
    }
    e.scene.add(group);
  }, [props.preview]);
  useEffect(() => {
    const e = engine.current;
    if (!e || !props.scene) return;
    const box = new T.Box3();
    let meshes = e.root.children.filter(
      (o: any) => o.userData.item.type !== 0 && o.visible,
    );
    const selected = meshes.filter(
      (o: any) =>
        o.userData.id === props.selected ||
        "body:" + o.userData.item.parent === props.selected,
    );
    if (props.fitRequest && selected.length) meshes = selected;
    for (const o of meshes) box.expandByObject(o);
    if (box.isEmpty()) return;
    const center = box.getCenter(new T.Vector3());
    const radius = Math.max(box.getSize(new T.Vector3()).length() / 2, 0.1);
    const direction = e.camera.position
      .clone()
      .sub(e.controls.target)
      .normalize();
    const distance =
      (radius / Math.sin(T.MathUtils.degToRad(e.camera.fov / 2))) * 1.35;
    e.controls.target.copy(center);
    e.camera.position.copy(center).addScaledVector(direction, distance);
    e.camera.near = Math.max(distance / 1000, 0.00001);
    e.camera.far = Math.max(distance * 100, 100);
    e.camera.updateProjectionMatrix();
  }, [props.fitRequest]);
  useEffect(() => {
    const e = engine.current;
    if (!e) return;
    const distance = e.camera.position.distanceTo(e.controls.target);
    const directions: Record<string, number[]> = {
      perspective: [2.1, -3, 1.6],
      front: [0, -1, 0],
      side: [1, 0, 0],
      top: [0, -0.001, 1],
    };
    const d = new T.Vector3()
      .fromArray(directions[props.viewDirection] || directions.perspective)
      .normalize();
    e.camera.position.copy(e.controls.target).addScaledVector(d, distance);
    e.controls.update();
  }, [props.viewDirection]);
  return (
    <div
      className={"viewport tool-" + props.tool}
      ref={host}
      data-testid="viewport"
    >
      <div className="view-corner">
        <span className="axis-z">Z</span>
        <span className="axis-y">Y</span>
        <span className="axis-x">X</span>
      </div>
    </div>
  );
}
function disposeGroup(group: T.Group) {
  for (const child of [...group.children]) {
    group.remove(child);
    child.traverse((o: any) => {
      o.geometry?.dispose();
      if (o.material)
        (Array.isArray(o.material) ? o.material : [o.material]).forEach(
          (m: any) => {
            m.map?.dispose();
            m.dispose();
          },
        );
    });
  }
}
