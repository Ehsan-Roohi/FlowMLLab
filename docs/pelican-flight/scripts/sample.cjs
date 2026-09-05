// Export deterministic SVG attributes without changing the retained HTML.
const fs=require('fs'),vm=require('vm'),path=require('path');
const root=path.resolve(__dirname,'..');
for(const name of ['first','medium','light']){
 const html=fs.readFileSync(path.join(root,'originals',name+'.html'),'utf8'),els={};
 for(const m of html.matchAll(/id="([^"]+)"/g))els[m[1]]={attrs:{},setAttribute(k,v){this.attrs[k]=String(v)},addEventListener(){}};
 const sandbox={document:{getElementById:id=>els[id],addEventListener(){},hidden:false},matchMedia:()=>({matches:false}),requestAnimationFrame(){}};
 let js=html.match(/<script>([\s\S]*?)<\/script>/)[1];
 js=js.replace(/\}\)\(\);\s*$/,`globalThis.capture=${name==='first'?'render':'draw'};})();`);
 vm.createContext(sandbox);vm.runInContext(js,sandbox);
 const frames=[];for(let i=0;i<360;i++){sandbox.capture(i/30);frames.push(Object.fromEntries(Object.entries(els).map(([k,v])=>[k,{...v.attrs}])))}
 fs.writeFileSync('/tmp/pelican-'+name+'.json',JSON.stringify(frames));
}
