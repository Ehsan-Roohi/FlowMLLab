// Record the existing static-elastic audit, without modifying any solver.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'..'),window={};
for(const id of ['extra-high','medium','light'])new Function('window',fs.readFileSync(path.join(root,'src',id+'.js'),'utf8'))(window);
new Function('window',fs.readFileSync(path.join(root,'src/adapters.js'),'utf8'))(window);
const audit=JSON.parse(fs.readFileSync(path.join(root,'results/audit.json')));
const output={scenario:'static-elastic',config:{omega:0,g:9.81,e:1},duration:60,sample_dt:1/30,solver_call_dt:1/2400,columns:['t_s','x_m','y_m','max_abs_energy_error_percent'],series:[]};
const energy=s=>.5*(s.vx*s.vx+s.vy*s.vy)+9.81*s.y;
for(const id of window.HexAdapters.ids){
 const m=window.HexAdapters.create(id,output.config),e0=energy(m.read());let peak=0;
 const points=[[0,m.read().x,m.read().y,0]];
 for(let k=1;k<=144000;k++){
  m.advance(1/2400);const s=m.read();peak=Math.max(peak,Math.abs(energy(s)-e0));
  if(k%80===0)points.push([+(k/2400).toFixed(6),+s.x.toFixed(8),+s.y.toFixed(8),+(100*peak/e0).toPrecision(10)]);
 }
 const expected=audit.results.find(r=>r.id===id).scenarios.find(s=>s.id==='static-elastic').max_absolute_invariant_drift_J;
 assert(Math.abs(peak-expected)<1e-12,'Replay must reproduce retained audit');
 output.series.push({id,initial_energy_J:e0,points});
}
fs.writeFileSync(path.join(root,'results/replay.js'),'window.hexReplay = '+JSON.stringify(output)+';\n');
console.log('Replay matches the stored audit for all three implementations.');
