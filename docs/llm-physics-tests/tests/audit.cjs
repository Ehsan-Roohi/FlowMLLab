// Independent black-box measurements: no solver gap, error, peak or energy counters.
const fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),window={};
const manifest=JSON.parse(fs.readFileSync(path.join(root,'manifest.json')));
for(const id of ['extra-high','medium','light']){
  const original=fs.readFileSync(path.join(root,'originals',id+'.html'),'utf8');
  const core=fs.readFileSync(path.join(root,'src',id+'.js'),'utf8');
  assert.equal(core,[...original.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)][0][1]);
  const recorded=manifest.find(m=>m.id===id),hash=text=>crypto.createHash('sha256').update(text).digest('hex');
  assert.equal(hash(original),recorded.original_sha256);assert.equal(hash(core),recorded.physics_sha256);
  new Function('window',core)(window);
}
new Function('window',fs.readFileSync(path.join(root,'src/adapters.js'),'utf8'))(window);
const {ids,create}=window.HexAdapters;
const dot=(a,b)=>a.x*b.x+a.y*b.y, r=.065, ap=Math.sqrt(3)/2;
const E=(s,g)=>.5*(s.vx*s.vx+s.vy*s.vy)+g*s.y;
const J=(s,w)=>E(s,0)-w*(s.x*s.vy-s.y*s.vx);
function penetration(s){let d=0;for(let i=0;i<6;i++){const a=s.theta+(i+.5)*Math.PI/3;d=Math.max(d,s.x*Math.cos(a)+s.y*Math.sin(a)+r-ap);}return d;}
function freeFlight(id){const m=create(id),s=m.read(),h=.05;m.advance(h);const q=m.read();return Math.hypot(q.x-s.x-s.vx*h,q.y-s.y-s.vy*h+.5*9.81*h*h);}
function collision(id,w){
  // Construct one exact collision at T, away from vertices, zero gravity.
  const T=.0005,end=.001,e=.85,a=.12+w*T+Math.PI/6;
  const n={x:Math.cos(a),y:Math.sin(a)},tan={x:-n.y,y:n.x};
  const p={x:(ap-r)*n.x+.2*tan.x,y:(ap-r)*n.y+.2*tan.y};
  const q={x:p.x+r*n.x,y:p.y+r*n.y},wall={x:-w*q.y,y:w*q.x};
  const v={x:wall.x+2*n.x+.3*tan.x,y:wall.y+2*n.y+.3*tan.y};
  const outgoing={x:v.x-(1+e)*2*n.x,y:v.y-(1+e)*2*n.y};
  const m=create(id,{omega:w,g:0,e});m.set({x:p.x-v.x*T,y:p.y-v.y*T,vx:v.x,vy:v.y});m.advance(end);const s=m.read();
  return {omega:w,velocity_error_m_s:Math.hypot(s.vx-outgoing.x,s.vy-outgoing.y),position_error_m:Math.hypot(s.x-p.x-outgoing.x*(end-T),s.y-p.y-outgoing.y*(end-T)),normal_law_residual_m_s:Math.abs(dot({x:s.vx-wall.x,y:s.vy-wall.y},n)+e*2),tangent_error_m_s:Math.abs(dot({x:s.vx-v.x,y:s.vy-v.y},tan))};
}
const scenarios=[
  {id:'default',omega:.7,g:9.81,e:.85},
  {id:'static-elastic',omega:0,g:9.81,e:1,invariant:'energy'},
  {id:'rotating-elastic',omega:3,g:0,e:1,invariant:'jacobi'},
  {id:'stationary-rest',omega:0,g:20,e:0},
  {id:'reverse',omega:-3,g:9.81,e:.85}
];
const report={generated_utc:new Date().toISOString(),node:process.version,method:'Black-box state audit; original counters unused',sample_dt_s:1/2400,duration_s:60,manifest:JSON.parse(fs.readFileSync(path.join(root,'manifest.json'))),results:[]};
for(const id of ids){
  const row={id,free_flight_error_m:freeFlight(id),isolated_collisions:[0,1.3,-2].map(w=>collision(id,w)),scenarios:[]};
  for(const c of scenarios){
    const m=create(id,c),start=m.read();let maxPen=0,maxDrift=0,minSpeed=Infinity;
    const invariant=s=>c.invariant==='jacobi'?J(s,c.omega):E(s,c.g),i0=invariant(start);
    for(let k=0;k<60*2400;k++){
      m.advance(1/2400);const s=m.read();assert([s.x,s.y,s.vx,s.vy,s.t].every(Number.isFinite));
      maxPen=Math.max(maxPen,penetration(s));if(c.invariant)maxDrift=Math.max(maxDrift,Math.abs(invariant(s)-i0));
      if(k>59*2400)minSpeed=Math.min(minSpeed,Math.hypot(s.vx,s.vy));
    }
    const final=m.read();row.scenarios.push({id:c.id,config:c,max_sampled_penetration_mm:maxPen*1000,invariant:c.invariant??null,max_absolute_invariant_drift_J:c.invariant?maxDrift:null,relative_invariant_drift:c.invariant?maxDrift/Math.max(Math.abs(i0),1e-12):null,final_speed_m_s:Math.hypot(final.vx,final.vy),last_second_min_speed_m_s:minSpeed,final});
  }
  report.results.push(row);console.log(id,JSON.stringify({freeFlight:row.free_flight_error_m,collisions:row.isolated_collisions,scenarios:row.scenarios.map(x=>({id:x.id,pen_mm:x.max_sampled_penetration_mm,drift:x.max_absolute_invariant_drift_J}))}));
}
fs.writeFileSync(path.join(root,'results/audit.json'),JSON.stringify(report,null,2)+'\n');
fs.writeFileSync(path.join(root,'results/data.js'),'window.auditData = '+JSON.stringify(report)+';\n');
console.log('Audit completed; results/audit.json written.');
