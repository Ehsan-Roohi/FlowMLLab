(()=>{
 const $=id=>document.getElementById(id),data=window.hexReplay;
 const names={'extra-high':'Extra High',medium:'Medium',light:'Light'},colors={'extra-high':'#1566a4',medium:'#684ba1',light:'#b34310'};
 let chosen=60,playing=false,stamp=null,width=800;
 const maxValue=Math.max(...data.series.map(s=>s.points[s.points.length-1][3]));
 const ceiling=Math.ceil(maxValue*1.1),index=()=>Math.min(1800,Math.round(chosen*30));
 $('replay-snapshots').innerHTML=data.series.map(s=>`<article class="snapshot"><h3>${names[s.id]}</h3><output id="replay-error-${s.id}" class="error-value"></output><span class="error-caption">maximum energy error</span><svg viewBox="-1.15 -1.15 2.3 2.3" role="img" aria-label="Recorded ${names[s.id]} trajectory"><g transform="scale(1,-1)"><polygon id="replay-wall-${s.id}" fill="none" stroke="#8c9ba8" stroke-width=".012"/><polyline id="replay-trail-${s.id}" fill="none" stroke="${colors[s.id]}" opacity=".35" stroke-width=".01"/><circle id="replay-ball-${s.id}" r=".065" fill="${colors[s.id]}"/></g></svg></article>`).join('');
 const fmt=n=>n===0?'0%':n<.001?n.toExponential(2)+'%':n.toFixed(2)+'%';
 function chart(){
  width=Math.max(280,$('energy-chart').getBoundingClientRect().width||800);
  const h=250,l=48,r=18,t=26,b=38,x=v=>l+v/60*(width-l-r),y=v=>h-b-v/ceiling*(h-t-b);
  let content=`<title>Maximum energy error, percent of initial energy</title><text x="${l}" y="16">Maximum energy error (%)</text>`;
  for(let v=0;v<=ceiling;v++){content+=`<line x1="${l}" x2="${width-r}" y1="${y(v)}" y2="${y(v)}" stroke="#d8e1e8"/><text x="${l-10}" y="${y(v)+4}" text-anchor="end">${v}</text>`;}
  for(let v=0;v<=60;v+=width<450?20:10)content+=`<text x="${x(v)}" y="${h-16}" text-anchor="middle">${v}</text>`;
  content+=`<text x="${width-r}" y="${h-1}" text-anchor="end">Time (s)</text>`;
  data.series.forEach(s=>{const dash=s.id==='medium'?'stroke-dasharray="5 4"':'';content+=`<polyline points="${s.points.map(p=>x(p[0]).toFixed(2)+','+y(p[3]).toFixed(3)).join(' ')}" stroke="${colors[s.id]}" stroke-width="2" fill="none" ${dash}/>`;});
  content+=`<line id="replay-cursor" x1="${x(chosen)}" x2="${x(chosen)}" y1="${t}" y2="${h-b}" stroke="#253c50" stroke-dasharray="3 4"/>`;
  $('energy-chart').setAttribute('viewBox',`0 0 ${width} ${h}`);$('energy-chart').innerHTML=content;
 }
 function update(){
  const i=index();$('replay-time').value=chosen;$('replay-time-label').textContent=(i/30).toFixed(1)+' s';
  const x=48+chosen/60*(width-66);$('replay-cursor').setAttribute('x1',x);$('replay-cursor').setAttribute('x2',x);
  const wall=Array.from({length:6},(_,j)=>[Math.cos(.12+j*Math.PI/3),Math.sin(.12+j*Math.PI/3)]).map(p=>p.join(',')).join(' ');
  data.series.forEach(s=>{const p=s.points[i];$('replay-error-'+s.id).textContent=fmt(p[3]);$('replay-wall-'+s.id).setAttribute('points',wall);$('replay-ball-'+s.id).setAttribute('cx',p[1]);$('replay-ball-'+s.id).setAttribute('cy',p[2]);$('replay-trail-'+s.id).setAttribute('points',s.points.slice(Math.max(0,i-90),i+1).map(q=>q[1]+','+q[2]).join(' '));});
 }
 function button(){$('replay-play').textContent=playing?'Pause replay':'Replay 60 s in 12 s';$('replay-play').setAttribute('aria-pressed',String(playing));}
 $('replay-time').addEventListener('input',()=>{playing=false;chosen=+$('replay-time').value;button();update();});
 $('replay-play').addEventListener('click',()=>{playing=!playing;if(playing&&chosen>=60)chosen=0;stamp=null;button();update();});
 function frame(now){if(playing&&stamp!==null&&!document.hidden){chosen=Math.min(60,chosen+Math.min((now-stamp)/1000,.1)*5);if(chosen>=60){playing=false;button();}update();}stamp=now;requestAnimationFrame(frame);}
 chart();update();button();if(typeof ResizeObserver!=='undefined')new ResizeObserver(()=>{chart();update();}).observe($('energy-chart'));requestAnimationFrame(frame);
})();
