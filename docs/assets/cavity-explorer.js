/* Saved scientific data only; no telemetry or client-side fitting. */
(() => {
'use strict';
const $=id=>document.getElementById(id), base='assets/cavity/';
const ids=['reynolds','quantity','comparison','region','show-streamlines'];
const bounds={full:[0,1,0,1],left:[0,.2,0,.2],right:[.8,1,0,.2],upper:[0,.2,.8,1]};
let manifest,generation=0;const cache=new Map();
const fmt=n=>Math.abs(n)<1e-12?'0':Math.abs(n)<.001?n.toExponential(1):Number(n.toPrecision(3)).toString();
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function getJSON(url){const response=await fetch(url);if(!response.ok)throw new Error('Data request failed: '+response.status);return response.json();}
function axes(b,xlabel='x / L',ylabel='y / L',grid=false){
 const [xmin,xmax,ymin,ymax]=b,x=v=>64+508*(v-xmin)/(xmax-xmin),y=v=>532-508*(v-ymin)/(ymax-ymin);
 let content='<rect x="64" y="24" width="508" height="508" fill="white"/>';
 if(grid)for(let i=0;i<=4;i++){const xv=xmin+(xmax-xmin)*i/4,yv=ymin+(ymax-ymin)*i/4;content+=`<path d="M${x(xv)} 24V532M64 ${y(yv)}H572" stroke="#dbe3e7" stroke-width="1"/>`;}
 let labels='<rect x="64" y="24" width="508" height="508" fill="none" stroke="#263c48" stroke-width="1.1"/>';
 for(let i=0;i<=4;i++){const xv=xmin+(xmax-xmin)*i/4,yv=ymin+(ymax-ymin)*i/4;labels+=`<text x="${x(xv)}" y="556" text-anchor="middle">${fmt(xv)}</text><text x="53" y="${y(yv)+5}" text-anchor="end">${fmt(yv)}</text>`;}
 labels+=`<text class="axis-label" x="318" y="589" text-anchor="middle">${xlabel}</text><text class="axis-label" transform="translate(18 278) rotate(-90)" text-anchor="middle">${ylabel}</text>`;
 return {x,y,content,labels};
}
function streamlinePaths(paths,ax,color,dashed){return paths.map(points=>{
 const d=points.map((p,i)=>(i?'L':'M')+ax.x(p[0]).toFixed(2)+','+ax.y(p[1]).toFixed(2)).join(' ');let arrow='';
 if(!dashed&&points.length>10){const i=Math.floor(points.length*.6),p=points[i],q=points[i+1],x=ax.x(p[0]),y=ax.y(p[1]),angle=Math.atan2(ax.y(q[1])-y,ax.x(q[0])-x)*180/Math.PI;arrow=`<path d="M-5,-2.5L0,0L-5,2.5" transform="translate(${x} ${y}) rotate(${angle})" fill="none" stroke="${color}" stroke-width="1"/>`;}
 return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${dashed?1.4:1}" ${dashed?'stroke-dasharray="5 4"':''} opacity=".92"/>`+arrow;
}).join('');}
function colorbar(scale,error){
 const colors=error?scale.error.colors:scale.colors,ticks=error?[{position:0,value:scale.error.min},{position:.5,value:0},{position:1,value:scale.error.max}]:scale.ticks;
 return `<div class="colorbar"><div class="colorbar-strip" style="background:linear-gradient(to right,${colors.join(',')})"></div><div class="colorbar-ticks">${ticks.map(t=>`<span style="left:${100*t.position}%">${fmt(t.value)}</span>`).join('')}</div></div><div class="colorbar-label">${error?'NN − CFD · ':''}${esc(scale.label)}<br>${error?'Linear scale':scale.normalization==='linear'?'Linear scale':'Symmetric-log scale'} · fixed across Re</div>`;
}
function fieldPanel(data,quantity,kind,overlay,b,streams){
 const ax=axes(b),id='field-'+kind,names={cfd:'CFD reference',nn:'POD-DeepONet',error:'Signed difference · NN − CFD'};let content=ax.content;
 if(quantity!=='streamlines'){const f=data.fields[quantity],[x0,x1,y0,y1]=f.extent;content+=`<g clip-path="url(#${id})"><image href="${base+f[kind]}" x="${ax.x(x0)}" y="${ax.y(y1)}" width="${ax.x(x1)-ax.x(x0)}" height="${ax.y(y0)-ax.y(y1)}" preserveAspectRatio="none"/></g>`;}
 if((streams||quantity==='streamlines'||overlay)&&kind!=='error'){const name=kind==='nn'?'POD-DeepONet':'CFD';content+=`<g clip-path="url(#${id})">`+streamlinePaths(data.streamlines[name],ax,'#111',false);if(overlay)content+=streamlinePaths(data.streamlines['POD-DeepONet'],ax,'#c65319',true);content+='</g>';}
 return `<article class="field-panel"><h3>${overlay?'CFD &amp; POD-DeepONet streamlines':names[kind]}</h3><svg viewBox="0 0 600 606" role="img" aria-label="${esc(names[kind]+' '+quantity+' field at Reynolds number '+data.re)}"><defs><clipPath id="${id}"><rect x="64" y="24" width="508" height="508"/></clipPath></defs>${content}${ax.labels}</svg>${quantity==='streamlines'?'':colorbar(manifest.scales[quantity],kind==='error')}</article>`;
}
function profile(data,component){
 const isU=component==='u',ax=axes(isU?[-1,1,0,1]:[0,1,-.7,.7],isU?'u / U_lid':'x / L',isU?'y / L':'v / U_lid',true);let content=ax.content;
 for(const name of ['CFD','POD-DeepONet']){const nn=name!=='CFD',values=data.profiles[name][component],d=values.map((v,i)=>(i?'L':'M')+ax.x(isU?v:data.coordinate[i]).toFixed(2)+','+ax.y(isU?data.coordinate[i]:v).toFixed(2)).join(' ');content+=`<path d="${d}" fill="none" stroke="${nn?'#c65319':'#006675'}" stroke-width="2" ${nn?'stroke-dasharray="7 4"':''}/>`;}
 const positions=data.ghia[isU?'y':'x'],values=data.ghia[component],flag=data.flags[component];
 for(let i=0;i<positions.length;i++){const x=ax.x(isU?values[i]:positions[i]),y=ax.y(isU?positions[i]:values[i]),title=`Ghia: ${isU?'y':'x'}/L=${positions[i]}, ${component}/U_lid=${values[i]}`;
 if(flag&&Math.abs(positions[i]-flag.position)<1e-5)content+=`<path d="M${x-4},${y-4}l8,8m-8,0l8,-8" stroke="#b01d32" stroke-width="2"><title>${esc(title+'; '+flag.reason)}</title></path>`;
 else content+=`<circle cx="${x}" cy="${y}" r="3.5" fill="white" stroke="#25333b" stroke-width="1.4"><title>${title}</title></circle>`;}
 return `<article class="profile-panel"><h4>${isU?'Vertical centreline · x/L = 0.5':'Horizontal centreline · y/L = 0.5'}</h4><svg viewBox="0 0 600 606" role="img" aria-label="${isU?'Horizontal velocity along the vertical':'Vertical velocity along the horizontal'} centreline; CFD, neural prediction and Ghia at Re ${data.re}">${content}${ax.labels}</svg></article>`;
}
async function render(){
 const ticket=++generation,re=$('reynolds').value,quantity=$('quantity').value,errorOption=$('comparison').querySelector('option[value="error"]');errorOption.disabled=quantity==='streamlines';
 if(errorOption.disabled&&$('comparison').value==='error')$('comparison').value='paired';
 const mode=$('comparison').value,region=$('region').value,streams=$('show-streamlines').checked;
 $('field-panels').setAttribute('aria-busy','true');$('cavity-loading').hidden=false;$('cavity-loading').textContent='Loading Re = '+Number(re).toLocaleString('en-US')+'…';
 try{
  if(!cache.has(re))cache.set(re,await getJSON(base+manifest.cases[re].file));if(ticket!==generation)return;
  const data=cache.get(re),b=bounds[region],overlay=mode==='overlay';if(data.re!==Number(re))throw new Error('Mismatched Reynolds number');
  $('field-panels').classList.toggle('single',overlay);$('field-panels').innerHTML=fieldPanel(data,quantity,'cfd',overlay,b,streams)+(overlay?'':fieldPanel(data,quantity,mode==='error'?'error':'nn',false,b,streams));
  $('case-heading').textContent='Re = '+data.re.toLocaleString('en-US')+' · steady square cavity';$('case-role').textContent='NN: '+data.role;$('overlay-legend').hidden=!overlay;
  $('field-description').textContent=quantity==='streamlines'?'Both methods use the same prescribed streamline seeds and integration settings. Zoom preserves matched spatial bounds.':quantity==='pressure'?'Pressure is recovered from velocity on interior nodes, with a zero-mean gauge for each field. Both methods share the pressure scale.':quantity==='vorticity'?'Signed vorticity from the velocity curl at interior nodes. No wall values are extrapolated.':'Native speed fields with one fixed scale for both methods and all seven Reynolds numbers.';
  if(mode==='error')$('field-description').textContent+=' The right panel uses its own signed, linear difference scale.';
  $('profile-panels').innerHTML=profile(data,'u')+profile(data,'v');$('profile-summary').textContent='Full-field velocity relative L₂ error of the frozen neural prediction against CFD: '+(100*data.velocity_relative_l2).toFixed(4)+'%. This is a '+data.role+' case, not an error against Ghia.';
  const flags=Object.entries(data.flags);$('ghia-note').textContent=flags.length?'All published Ghia samples are retained. The × marker identifies '+flags.map(([c,f])=>`${c}: ${f.reason}`).join('; ')+'. The source record provides both published-sample and separately screened scores.':'All 17 published Ghia samples per component are shown. Curves use all 513 native centreline nodes; reference points are not training data.';
  $('profile-download').href=base+'re'+re+'-profiles.csv';
  const url=new URL(location.href);for(const [key,value] of Object.entries({re,field:quantity,view:mode,region,streams:streams?'1':'0'}))url.searchParams.set(key,value);history.replaceState(null,'',url);url.hash='cavity';$('share-view').href=url.href;$('share-view').target='_blank';$('share-view').rel='noopener noreferrer';
  $('cavity-loading').hidden=true;$('field-panels').setAttribute('aria-busy','false');
 }catch(error){if(ticket!==generation)return;$('cavity-loading').textContent='This case could not be loaded. Select it again to retry, or use the linked static results.';$('overlay-legend').hidden=true;for(const id of ['field-panels','profile-panels'])$(id).innerHTML='';for(const id of ['case-heading','case-role','profile-summary','ghia-note','field-description'])$(id).textContent='';$('field-panels').setAttribute('aria-busy','false');console.error(error);}
}
async function init(){try{manifest=await getJSON(base+'manifest.json');const q=new URLSearchParams(location.search);
 for(const [id,key] of [['reynolds','re'],['quantity','field'],['comparison','view'],['region','region']]){const value=q.get(key);if([...$(id).options].some(o=>o.value===value))$(id).value=value;}
 if(q.has('streams'))$('show-streamlines').checked=q.get('streams')!=='0';for(const id of ids)$(id).addEventListener('change',render);$('cavity-controls').disabled=false;await render();
 }catch(error){$('cavity-loading').textContent='The interactive data could not be loaded. Reload to retry, or follow the static results and notebook links below.';console.error(error);}}
init();
})();
