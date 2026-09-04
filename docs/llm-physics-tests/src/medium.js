
window.makeHexMedium = function(options={}) {
  const c=Object.assign({w:0.7,g:9.81,e:0.85},options);
  const s={x:0.08,y:0.36,vx:0.72,vy:0.1,t:0,maxPen:0,error:0,last:null,hits:0};
  const r=0.065,ap=Math.sqrt(3)/2;
  function face(i,t) {const a=0.12+c.w*t+(i+0.5)*Math.PI/3;return {x:Math.cos(a),y:Math.sin(a)};}
  function gap(i,dt=0) {const n=face(i,s.t+dt);return ap-r-n.x*(s.x+s.vx*dt)-n.y*(s.y+s.vy*dt-0.5*c.g*dt*dt);}
  function drift(dt) {s.x+=s.vx*dt;s.y+=s.vy*dt-0.5*c.g*dt*dt;s.vy-=c.g*dt;s.t+=dt;}
  function hit(i,e) {
    const n=face(i,s.t),qx=s.x+r*n.x,qy=s.y+r*n.y;
    const wx=-c.w*qy,wy=c.w*qx,un=(s.vx-wx)*n.x+(s.vy-wy)*n.y;
    if(un<=0)return;
    s.vx-=(1+e)*un*n.x;s.vy-=(1+e)*un*n.y;
    const after=(s.vx-wx)*n.x+(s.vy-wy)*n.y;
    s.error=Math.max(s.error,Math.abs(after+e*un));s.last={before:un,after,e};s.hits++;
  }
  function project() {
    for(let k=0;k<12;k++){
      let corrected=false;
      for(let i=0;i<6;i++) {const d=gap(i);s.maxPen=Math.max(s.maxPen,-d);if(d<0){const n=face(i,s.t);s.x+=n.x*(d-1e-12);s.y+=n.y*(d-1e-12);hit(i,0);corrected=true;}}
      if(!corrected)break;
    }
  }
  function substep(dt) {
    let left=dt;
    for(let events=0;events<10 && left>1e-12;events++){
      let at=left,which=-1;
      for(let i=0;i<6;i++)if(gap(i,left)<0){
        let lo=0,hi=left;
        if(gap(i)>0){for(let j=0;j<35;j++){const mid=(lo+hi)/2;if(gap(i,mid)>0)lo=mid;else hi=mid;}}
        else hi=0;
        if(hi<=at){at=hi;which=i;}
      }
      if(which<0){drift(left);left=0;break;}
      drift(at);left-=at;
      hit(which,c.e);
      const n=face(which,s.t);s.x-=n.x*1e-11;s.y-=n.y*1e-11;
    }
    if(left>0)drift(left);
    project();
  }
  function advance(dt) {
    while(dt>1e-12){const speed=Math.hypot(s.vx,s.vy)+Math.abs(c.w);const h=Math.min(dt,1/2400,0.001/Math.max(speed,1));substep(h);dt-=h;}
  }
  return {s,c,r,face,gap,hit,advance};
};
