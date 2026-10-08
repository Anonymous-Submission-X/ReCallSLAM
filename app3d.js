import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const NAMES = {
  ours:"ReCall-SLAM", lingbot:"LingBot-Map", abot:"ABot-Recon",
  scarf:"ScaRF-SLAM", vggt_rgbd:"VGGT-SLAM 2.0-D",
  vggt_pi3x:"VGGT-SLAM 2.0-Pi3X", twodgs:"2DGS-SLAM", profusion:"PROFusion"
};
const LONG_METHODS = ["lingbot","abot","scarf","vggt_rgbd","vggt_pi3x"];
const FAST_METHODS = [...LONG_METHODS,"twodgs","profusion"];
const LONG_SCENES = [
  {id:"build_2_f3-4",name:"Multi-floor corridor",frames:4300},
  {id:"highschool",name:"Classrooms",frames:27002},
  {id:"bupt",name:"Tiered lecture hall",frames:30079}
];
const FAST_SCENES = ["meeting_room","gym","office","studio","lab","stairwell","lounge_1","lounge_2","workshop_1","workshop_2","apartment_1","apartment_2"];
const OXFORD_SCENES = [
  "2024-03-18-christ-church-02","2024-03-12-keble-college-04",
  "2024-03-13-observatory-quarter-01","2024-03-14-blenheim-palace-01"
];
const states = { long:{scene:1,method:"vggt_rgbd"}, fast:{scene:1,method:"vggt_rgbd"}, oxford:{scene:0,method:"vggt_rgbd"} };
const manifests = new Map();
const cloudCache = new Map();
const viewers = {};
// Use the bright portion of viridis so the whole trajectory reads on white.
const TRACK_STOPS = [0x2a788e,0x1e9c89,0x44bf70,0x9bd93c,0xfde725].map(value => new THREE.Color(value));
function trackColor(t) {
  const scaled=Math.max(0,Math.min(1,t))*(TRACK_STOPS.length-1);
  const index=Math.min(TRACK_STOPS.length-2,Math.floor(scaled));
  return TRACK_STOPS[index].clone().lerp(TRACK_STOPS[index+1],scaled-index);
}

async function manifest(path) {
  if (!manifests.has(path)) manifests.set(path, fetch(path).then(response => {
    if (!response.ok) throw new Error(`Could not load ${path}`);
    return response.json();
  }));
  return manifests.get(path);
}
function longManifest() { return manifest(`assets/3d/long/${LONG_SCENES[states.long.scene].id}.json`); }
function fastManifest() { return manifest("assets/3d/fastcamo.json"); }
function oxfordManifest() { return manifest("assets/3d/oxford.json"); }

async function unpackCloud(path) {
  if (!cloudCache.has(path)) cloudCache.set(path, (async () => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Could not load ${path}`);
    if (!window.DecompressionStream) throw new Error("This browser needs gzip stream support");
    const buffer = await new Response(response.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    const data = new DataView(buffer);
    if ([0,1,2,3].some((i) => data.getUint8(i) !== [82,67,80,49][i])) throw new Error("Point cloud format mismatch");
    const count = data.getUint32(4,true);
    if (buffer.byteLength !== 32 + 9*count) throw new Error("Point cloud length mismatch");
    const minimum = [8,12,16].map(offset => data.getFloat32(offset,true));
    const extent = [20,24,28].map(offset => data.getFloat32(offset,true));
    const position = new Float32Array(count*3), color = new Uint8Array(count*3);
    for (let i=0, offset=32; i<count; i++,offset+=9) {
      for (let axis=0; axis<3; axis++) {
        position[i*3+axis] = minimum[axis] + extent[axis]*data.getUint16(offset+axis*2,true)/65535;
        color[i*3+axis] = data.getUint8(offset+6+axis);
      }
    }
    return {position,color,radius:Math.hypot(...extent)/2};
  })());
  return cloudCache.get(path);
}

class ViewerPair {
  constructor(id, showTrajectory) {
    this.element = document.getElementById(id);
    this.showTrajectory = showTrajectory;
    this.views = [...this.element.querySelectorAll(".viewer-slot")].map(slot => {
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf7f9fa);
      const camera = new THREE.PerspectiveCamera(45,1,.01,100000);
      camera.up.set(0,0,1);
      const renderer = new THREE.WebGLRenderer({antialias:false,alpha:false,powerPreference:"low-power"});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.5));
      slot.querySelector(".viewer-canvas").append(renderer.domElement);
      const control = new OrbitControls(camera,renderer.domElement);
      control.enableDamping = false;
      control.enablePan = true;
      return {slot,scene,camera,renderer,control,points:null,tracks:[],radius:1,serial:0};
    });
    this.views.forEach((view,index) => view.control.addEventListener("change",() => {
      if (this.syncing) return;
      this.syncing = true;
      const other=this.views[1-index];
      const direction=view.camera.position.clone().sub(view.control.target).normalize();
      const distance=view.camera.position.distanceTo(view.control.target);
      const scaled=distance*Math.max(other.radius,1)/Math.max(view.radius,1);
      other.control.target.copy(view.control.target);
      other.camera.position.copy(other.control.target).addScaledVector(direction,scaled);
      other.camera.lookAt(other.control.target);
      this.syncing=false;
      this.render();
    }));
    this.observer = new ResizeObserver(() => this.resize());
    this.views.forEach(view => this.observer.observe(view.slot));
    this.resize();
  }
  resize() {
    for (const view of this.views) {
      view.camera.aspect=Math.max(1,view.slot.clientWidth)/Math.max(1,view.slot.clientHeight);
      view.camera.updateProjectionMatrix();
      view.renderer.setSize(Math.max(1,view.slot.clientWidth),Math.max(1,view.slot.clientHeight),false);
    }
    this.render();
  }
  render() { this.views.forEach(view => view.renderer.render(view.scene,view.camera)); }
  clear(index) {
    const view=this.views[index];
    if (view.points) {
      view.scene.remove(view.points);
      view.points.geometry.dispose();
      view.points.material.dispose();
      view.points=null;
    }
    view.tracks.forEach(track => {
      view.scene.remove(track);
      track.geometry.dispose();
      track.material.dispose();
    });
    view.tracks=[];
  }
  async set(index,entry) {
    const view=this.views[index], serial=++view.serial;
    const status=view.slot.querySelector(".viewer-state");
    this.clear(index);
    if (!entry || entry.status!=="ready") {
      status.textContent="3D files pending transfer";
      view.radius=1;
      this.render();
      return;
    }
    status.textContent="Loading 3D…";
    try {
      const cloud=await unpackCloud(entry.cloud);
      if (view.serial!==serial) return;
      const geometry=new THREE.BufferGeometry();
      geometry.setAttribute("position",new THREE.BufferAttribute(cloud.position,3));
      geometry.setAttribute("color",new THREE.Uint8BufferAttribute(cloud.color,3,true));
      const pointSize=this.element.id==="fast-viewers" ? 1.5 : this.element.id==="oxford-viewers" ? 1.35 : 1;
      view.points=new THREE.Points(geometry,new THREE.PointsMaterial({size:pointSize,sizeAttenuation:false,vertexColors:true}));
      view.scene.add(view.points);
      view.radius=Math.max(cloud.radius,1);
      if (this.showTrajectory && entry.trajectory?.length>1) {
        const step=Math.max(1,Math.ceil(entry.trajectory.length/700));
        const path=entry.trajectory.filter((_,i) => i%step===0 || i===entry.trajectory.length-1).map(p => new THREE.Vector3(...p));
        const curve=new THREE.CatmullRomCurve3(path,false,"centripetal");
        const segmentCount=Math.max(2,path.length*2);
        const radius=Math.max(.025,Math.min(view.radius*.0042,.42));
        const radialSegments=6;
        const tubeGeometry=new THREE.TubeGeometry(curve,segmentCount,radius,radialSegments,false);
        const colors=new Float32Array(tubeGeometry.attributes.position.count*3);
        for (let i=0; i<tubeGeometry.attributes.position.count; i++) {
          const t=Math.floor(i/(radialSegments+1))/segmentCount;
          const color=trackColor(t);
          colors[i*3]=color.r;
          colors[i*3+1]=color.g;
          colors[i*3+2]=color.b;
        }
        tubeGeometry.setAttribute("color",new THREE.BufferAttribute(colors,3));
        const tube=new THREE.Mesh(tubeGeometry,
          new THREE.MeshBasicMaterial({vertexColors:true,depthTest:false,depthWrite:false}));
        tube.renderOrder=9;
        view.tracks.push(tube);
        view.scene.add(tube);
      }
      status.textContent="";
      this.fit();
      this.render();
    } catch(error) {
      if (view.serial!==serial) return;
      status.textContent="3D load failed";
      console.error(error);
    }
  }
  fit() {
    for (const view of this.views) {
      const radius=Math.max(1,view.radius);
      const isLong=this.element.id==="long-viewers";
      const distance=radius*(isLong ? (view.camera.aspect<1.2?2.3:1.85)
        : (view.camera.aspect<1.2?2.8:2.25));
      view.camera.near=Math.max(.005,radius/2000);
      view.camera.far=radius*100;
      view.camera.position.set(distance*.36,-distance*.75,distance*.56);
      view.camera.lookAt(0,0,0);
      view.camera.updateProjectionMatrix();
      view.control.target.set(0,0,0);
      view.control.minDistance=radius*.1;
      view.control.maxDistance=radius*30;
      view.control.update();
    }
  }
}

function fillSelect(id,methods,selected,onChange,pending=() => false) {
  const select=document.getElementById(id);
  select.replaceChildren(...methods.map(method => {
    const option=document.createElement("option");
    option.value=method;
    option.textContent=NAMES[method]+(pending(method)?" · files pending":"");
    return option;
  }));
  select.value=selected;
  select.onchange=event => onChange(event.target.value);
}
function setVideo(video,src,poster) {
  const source=video.querySelector("source");
  if (source.getAttribute("src")===src) return;
  video.pause();
  video.poster=poster;
  source.src=src;
  video.load();
  video.playbackRate=2;
  video.play().catch(()=>{});
}
function coverageText(entry,total) {
  if (!entry || entry.status!=="ready") return "Files pending";
  const [first,last]=entry.source_frame_range||[0,total-1];
  if (first>1 || last<total-2) return `Frames ${first.toLocaleString()}–${last.toLocaleString()} · partial`;
  if (entry.frames<total*.75) return `${entry.frames.toLocaleString()} sampled poses`;
  return `${total.toLocaleString()} frames`;
}
function renderLongCards() {
  const root=document.getElementById("long-scenes");
  root.replaceChildren(...LONG_SCENES.map((scene,index) => {
    const button=document.createElement("button");
    button.className="scene-card";
    button.type="button";
    button.setAttribute("aria-pressed",String(index===states.long.scene));
    button.style.setProperty("--length",`${(scene.frames/LONG_SCENES[2].frames*100).toFixed(1)}%`);
    button.innerHTML=`<img src="assets/videos/${scene.id}/ours.webp" alt="" loading="lazy"><span class="scene-info"><strong>${scene.name}</strong><span class="scene-count"><b>${scene.frames.toLocaleString()}</b><small>frames</small></span><span class="scene-length" aria-hidden="true"><i></i></span></span>`;
    button.onclick=() => {states.long.scene=index; updateLong();};
    return button;
  }));
}
function renderBaselineVideos(data) {
  const scene=LONG_SCENES[states.long.scene], root=document.getElementById("baseline-videos");
  root.replaceChildren(...LONG_METHODS.map(method => {
    const figure=document.createElement("figure");
    const coverage=coverageText(data.methods[method],scene.frames);
    figure.className="mini-video";
    figure.innerHTML=`<figcaption>${NAMES[method]}</figcaption><video muted loop autoplay playsinline preload="metadata" poster="assets/videos/${scene.id}/${method}.webp"><source src="assets/videos/${scene.id}/${method}.mp4" type="video/mp4"></video><div class="coverage ${coverage.includes("partial")?"partial":""}">${coverage}</div>`;
    return figure;
  }));
  root.querySelectorAll("video").forEach(video => {
    video.playbackRate=2;
    video.play().catch(()=>{});
  });
}
async function updateLong() {
  const scene=LONG_SCENES[states.long.scene];
  renderLongCards();
  document.getElementById("long-video-scene-name").textContent=scene.name;
  document.getElementById("long-video-frame-count").textContent=scene.frames.toLocaleString();
  setVideo(document.getElementById("long-ours-video"),`assets/videos/${scene.id}/ours.mp4`,`assets/videos/${scene.id}/ours.webp`);
  const data=await longManifest();
  if (scene.id!==LONG_SCENES[states.long.scene].id) return;
  renderBaselineVideos(data);
  const entry=data.methods[states.long.method];
  document.getElementById("long-note").textContent=coverageText(entry,scene.frames).includes("partial")
    ? `${NAMES[states.long.method]} returns only ${coverageText(entry,scene.frames)}.`
    : "";
  if (viewers.long) await Promise.all([viewers.long.set(0,data.methods.ours),viewers.long.set(1,entry)]);
}
function makeDots(id,scenes,state,onSelect,label) {
  const root=document.getElementById(id);
  root.replaceChildren(...scenes.map((scene,index) => {
    const button=document.createElement("button");
    button.type="button";
    button.setAttribute("aria-label",`${label} ${index+1}: ${scene.replaceAll('_',' ')}`);
    button.setAttribute("aria-current",String(index===state.scene));
    button.title=scene.replaceAll('_',' ');
    button.onclick=() => {state.scene=index;onSelect();};
    return button;
  }));
}
async function updateFast() {
  makeDots("fast-dots",FAST_SCENES,states.fast,updateFast,"Scene");
  const data=await fastManifest(), scene=FAST_SCENES[states.fast.scene];
  const ours=data.scenes[scene].ours, entry=data.scenes[scene][states.fast.method];
  document.getElementById("fast-note").textContent=entry.frames<ours.frames*.98
    ? `${NAMES[states.fast.method]}: ${entry.frames.toLocaleString()} / ${ours.frames.toLocaleString()} poses · no GT trajectory`
    : "";
  if (viewers.fast) await Promise.all([viewers.fast.set(0,ours),viewers.fast.set(1,entry)]);
}
function plotPath(points,bounds) {
  if (!points || points.length<2) return "";
  const [minX,maxX,minY,maxY]=bounds;
  const scale=Math.min(900/Math.max(maxX-minX,.001),260/Math.max(maxY-minY,.001));
  const cx=500-(minX+maxX)*scale/2, cy=165+(minY+maxY)*scale/2;
  return points.map((p,i)=>`${i?"L":"M"}${(cx+p[0]*scale).toFixed(1)},${(cy-p[1]*scale).toFixed(1)}`).join(" ");
}
function coloredPlotPaths(points,bounds) {
  if (!points || points.length<2) return "";
  const stride=Math.max(1,Math.ceil((points.length-1)/48));
  const paths=[];
  for (let start=0; start<points.length-1; start+=stride) {
    const segment=points.slice(start,Math.min(points.length,start+stride+1));
    paths.push(`<path d="${plotPath(segment,bounds)}" fill="none" stroke="${trackColor(start/(points.length-1)).getStyle()}" stroke-width="3.2"/>`);
  }
  return paths.join("");
}
function drawOxfordPlots(sceneData,method) {
  const gt=sceneData.plot.gt, ours=sceneData.plot.ours, baseline=sceneData.plot[method]||[];
  const all=[...gt,...ours,...baseline];
  const xs=all.map(p=>p[0]), ys=all.map(p=>p[1]);
  const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const xpad=Math.max(3,(maxX-minX)*.07),ypad=Math.max(3,(maxY-minY)*.07);
  const bounds=[minX-xpad,maxX+xpad,minY-ypad,maxY+ypad];
  for (const [id,methodPath] of [["oxford-ours-trajectory",ours],["oxford-baseline-trajectory",baseline]]) {
    const svg=document.getElementById(id);
    svg.innerHTML=`<path d="${plotPath(gt,bounds)}" fill="none" stroke="#98a1a8" stroke-width="2.6"/>${coloredPlotPaths(methodPath,bounds)}${methodPath.length?"":"<text x='500' y='170' text-anchor='middle' fill='#7c858d' font-size='25'>Trajectory files pending transfer</text>"}`;
  }
  document.getElementById("oxford-trajectory-method").textContent=NAMES[method];
}
async function updateOxford() {
  makeDots("oxford-dots",OXFORD_SCENES,states.oxford,updateOxford,"Route");
  const data=await oxfordManifest(),scene=OXFORD_SCENES[states.oxford.scene];
  const sceneData=data.scenes[scene],method=states.oxford.method,entry=sceneData.methods[method];
  fillSelect("oxford-method",LONG_METHODS,method,value=>{states.oxford.method=value;updateOxford();},key=>sceneData.methods[key].status!=="ready");
  drawOxfordPlots(sceneData,method);
  document.getElementById("oxford-note").textContent=entry.status!=="ready"
    ? `${NAMES[method]}: point cloud and trajectory pending transfer.`
    : entry.frames<sceneData.full_sequence_frames*.98
      ? `${NAMES[method]}: ${entry.frames.toLocaleString()} / ${sceneData.full_sequence_frames.toLocaleString()} poses.`
      : `${sceneData.full_sequence_frames.toLocaleString()}-frame complete route`;
  if (viewers.oxford) await Promise.all([viewers.oxford.set(0,sceneData.methods.ours),viewers.oxford.set(1,entry)]);
}
function lazyViewer(group,showTrajectory,update) {
  const target=document.getElementById(`${group}-viewers`);
  const observer=new IntersectionObserver(entries=>{
    if (!entries.some(entry=>entry.isIntersecting)) return;
    observer.disconnect();
    try { viewers[group]=new ViewerPair(target.id,showTrajectory); update(); }
    catch(error) { target.querySelectorAll(".viewer-state").forEach(node=>node.textContent="3D viewer unavailable"); console.error(error); }
  },{rootMargin:"400px"});
  observer.observe(target);
}
fillSelect("long-method",LONG_METHODS,states.long.method,value=>{states.long.method=value;updateLong();});
document.getElementById("long-ours-video").playbackRate=2;
fillSelect("fast-method",FAST_METHODS,states.fast.method,value=>{states.fast.method=value;updateFast();});
for (const [name,scenes,update] of [["fast",FAST_SCENES,updateFast],["oxford",OXFORD_SCENES,updateOxford]]) {
  document.getElementById(`${name}-prev`).onclick=()=>{states[name].scene=(states[name].scene-1+scenes.length)%scenes.length;update();};
  document.getElementById(`${name}-next`).onclick=()=>{states[name].scene=(states[name].scene+1)%scenes.length;update();};
}
lazyViewer("long",true,updateLong);
lazyViewer("fast",false,updateFast);
lazyViewer("oxford",true,updateOxford);
updateLong();
updateFast();
updateOxford();
