const fs=require('fs'),path=require('path'),assert=require('assert');
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(root+'/index.html','utf8');
for(const m of html.matchAll(/(?:src|href)="([^"]+)"/g)){if(!/^(https?:|#)/.test(m[1]))assert(fs.existsSync(path.join(root,m[1])),m[1]);}
const nodes={};function ingest(s){for(const m of s.matchAll(/\bid="([^"]+)"/g)){
 nodes[m[1]]={value:m[1]==='scenario'?'default':'',getBoundingClientRect(){return {width:736}},setAttribute(k,v){this[k]=v},addEventListener(k,f){this[k]=f},set innerHTML(v){this.content=v;ingest(v)},get innerHTML(){return this.content}};
}}
ingest(html);const window={},document={hidden:false,getElementById:id=>{assert(nodes[id],'Missing id '+id);return nodes[id]}};
let callbacks=[];const tick=t=>{const old=callbacks;callbacks=[];old.forEach(f=>f(t));};
for(const file of ['src/extra-high.js','src/medium.js','src/light.js','src/adapters.js','results/data.js','app.js','results/replay.js','replay.js'])new Function('window','document','matchMedia','requestAnimationFrame',fs.readFileSync(root+'/'+file,'utf8'))(window,document,()=>({matches:false}),f=>{callbacks.push(f)});
assert.equal(nodes['replay-time-label'].textContent,'60.0 s');assert.equal(nodes['replay-error-light'].textContent,'3.88%');
nodes['replay-time'].value=0;nodes['replay-time'].input();assert.equal(nodes['replay-error-light'].textContent,'0%');
nodes['replay-play'].click();tick(0);tick(100);assert.equal(nodes['replay-time-label'].textContent,'0.5 s');nodes['replay-play'].click();
nodes['replay-time'].value=60;nodes['replay-time'].input();assert.equal(nodes['replay-error-light'].textContent,'3.88%');
nodes['live-details'].open=true;nodes.play.click();tick(200);tick(240);assert.notEqual(+nodes['light-ball'].cx,.08);nodes.play.click();assert.equal(nodes.play.textContent,'Play');
for(const c of ['static-elastic','rotating-elastic','stationary-rest','reverse','default']){nodes.scenario.value=c;nodes.scenario.change();assert.equal(nodes.time.textContent,'0.0 s');nodes.reset.click();}
assert(nodes.results.innerHTML.includes('1.47e-1'));assert(nodes.finding.textContent.includes('3.88%'));
console.log('All local assets resolve; scripts parse; animation, pause, five scenarios, restart and results pass DOM smoke checks.');
