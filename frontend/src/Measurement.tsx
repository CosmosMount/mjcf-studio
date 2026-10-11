import { useState } from 'react';
import * as T from 'three';
import type { Scene, State } from './types';

export default function Measurement({scene,state,selected,pair,onPair,onAlign,onSave,onZero,busy}: {
 scene: Scene; state: State|null; selected:string; pair:string[]; onPair:(p:string[])=>void;
 onAlign:(v:any)=>Promise<void>; onSave:(name:string)=>void; onZero:(id:string)=>void; busy:boolean;
}) {
 const [frame,setFrame]=useState('world'), [excluded,setExcluded]=useState<string[]>([]), [name,setName]=useState('horizontal_reference');
 const joints=scene.joints;
 const adjustable=joints.filter(j=>j.type===2||j.type===3);
 const drivers=adjustable.filter(j=>!excluded.includes(j.name)).map(j=>j.name);
 const a=state?.joints?.find(j=>j.name===pair[0]), b=state?.joints?.find(j=>j.name===pair[1]);
 const base=state?.bodies?.find(b=>b.name===frame);
 let delta=a&&b ? new T.Vector3(...b.pos).sub(new T.Vector3(...a.pos)):null;
 if(delta && frame!=='world') {
  if(base) {const q=base.quat; delta.applyQuaternion(new T.Quaternion(q[1],q[2],q[3],q[0]).invert());}
  else delta=null;
 }
 const distance=delta?.length()||0;
 const elevation=delta&&distance>1e-9 ? Math.atan2(delta.z,Math.hypot(delta.x,delta.y))*180/Math.PI:null;
 const valid=!!a&&!!b&&pair[0]!==pair[1]&&distance>1e-9;
 return <section className="measurement">
  <h3>两关节测量与配合</h3>
  {[0,1].map(i=><div key={i}><label>关节 {i===0?'A':'B'}<select aria-label={`测量关节 ${i===0?'A':'B'}`} value={pair[i]||''} onChange={e=>{const p=[...pair];p[i]=e.target.value;onPair(p);}}><option value="">请选择</option>{joints.map(j=><option key={j.id} value={j.name}>{j.name}</option>)}</select></label><button disabled={!selected.startsWith('joint:')} onClick={()=>{const p=[...pair];p[i]=selected.slice(6);onPair(p);}}>使用选中为 {i===0?'A':'B'}</button></div>)}
  <label>测量坐标系<select aria-label="测量坐标系" value={frame} onChange={e=>setFrame(e.target.value)}>{scene.bodies.map(b=><option key={b.id} value={b.name}>{b.name==='world'?'世界':b.name}</option>)}</select></label>
  <output data-testid="joint-measurement">{delta ? <>中心距离 {(distance*1000).toFixed(3)} mm<br/>ΔX / ΔY / ΔZ：{delta.toArray().map(v=>(v*1000).toFixed(3)).join(' / ')} mm<br/>连线仰角 {elevation===null?'未定义':elevation.toFixed(4)+'°'}</>:'请选择两个关节'}</output>
  <details><summary>允许配合调整的关节（{drivers.length}）</summary>{adjustable.map(j=><label key={j.id}><input type="checkbox" aria-label={`配合调整 ${j.name}`} checked={drivers.includes(j.name)} onChange={e=>setExcluded(e.target.checked?excluded.filter(n=>n!==j.name):[...excluded,j.name])}/>{j.name}</label>)}</details>
  <p>水平 = 所选坐标系中 ΔZ 为零。保持闭链、关节限位与肌腱行程；不做碰撞求解。</p>
  <button disabled={busy||!valid||!drivers.length} onClick={()=>onAlign({first:pair[0],second:pair[1],frame,drivers})}>连线水平配合</button>
  <label>参考姿态名称<input aria-label="参考姿态名称" value={name} onChange={e=>setName(e.target.value)}/></label>
  <button disabled={busy||!valid||!name.trim()||Math.abs(delta?.z||0)>1e-6} onClick={()=>onSave(name)}>保存水平参考姿态</button>
  <button disabled={busy||!selected.startsWith('joint:')} onClick={()=>onZero(selected)}>标定选中关节零点…</button>
 </section>;
}
