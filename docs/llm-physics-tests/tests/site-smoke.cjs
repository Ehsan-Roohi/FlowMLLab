const fs=require('fs'),path=require('path'),assert=require('assert');
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(root+'/index.html','utf8');
for(const m of html.matchAll(/(?:src|href)="([^"]+)"/g)){if(!/^(https?:|#)/.test(m[1]))assert(fs.existsSync(path.join(root,m[1])),m[1]);}
const nodes={};function ingest(s){for(const m of s.matchAll(/\bid="([^"]+)"/g)){
 assert(!nodes[m[1]],'Duplicate id '+m[1]);nodes[m[1]]={value:m[1]==='scenario'?'default':'',setAttribute(k,v){this[k]=v},addEventListener(k,f){this[k]=f},set innerHTML(v){this.content=v;ingest(v)},get innerHTML(){return this.content}};
}}
ingest(html);const window={},document={hidden:false,getElementById:id=>{assert(nodes[id],'Missing id '+id);return nodes[id]}};
let raf;
for(const file of ['src/extra-high.js','src/medium.js','src/light.js','src/adapters.js','results/data.js','app.js'])new Function('window','document','matchMedia','requestAnimationFrame',fs.readFileSync(root+'/'+file,'utf8'))(window,document,()=>({matches:false}),f=>{raf=f});
raf(0);raf(40);assert.notEqual(+nodes['light-ball'].cx,.08);nodes.play.click();assert.equal(nodes.play.textContent,'Play');
for(const c of ['static-elastic','rotating-elastic','stationary-rest','reverse','default']){nodes.scenario.value=c;nodes.scenario.change();assert.equal(nodes.time.textContent,'0.0 s');nodes.reset.click();}
assert(nodes.results.innerHTML.includes('1.47e-1'));assert(nodes.finding.textContent.includes('3.88%'));
console.log('All local assets resolve; scripts parse; animation, pause, five scenarios, restart and results pass DOM smoke checks.');
