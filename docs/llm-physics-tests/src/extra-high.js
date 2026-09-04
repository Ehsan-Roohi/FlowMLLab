
(function(global){
'use strict';
const R=1, radius=.065, A=Math.sqrt(3)/2, EPS=1e-10;
function createSimulation(options={}) {
  const cfg=Object.assign({omega:.7,g:9.81,e:.85,step:1/480},options);
  const s={x:.08,y:.36,vx:.72,vy:.1,theta:.12,t:0,collisions:0,peakPen:0,wallWork:0,loss:0,last:null,fallbacks:0};
  const energy=()=>.5*(s.vx*s.vx+s.vy*s.vy)+cfg.g*s.y;
  s.E0=energy();
  let sleeping=false;
  let ax=0,ay=-cfg.g;
  function normal(i,theta=s.theta){const a=theta+(i+.5)*Math.PI/3;return {x:-Math.cos(a),y:-Math.sin(a)};}
  function gap(i,t=0){const n=normal(i,s.theta+cfg.omega*t);return n.x*(s.x+s.vx*t+.5*ax*t*t)+n.y*(s.y+s.vy*t+.5*ay*t*t)+A-radius;}
  function support(){
    ax=0;ay=-cfg.g;
    if(Math.abs(cfg.omega)>1e-12)return;
    // A stationary smooth wall supplies a normal reaction during sustained contact.
    for(let pass=0;pass<32;pass++){
      let changed=false;
      for(let i=0;i<6;i++){
        const n=normal(i),vn=s.vx*n.x+s.vy*n.y,an=ax*n.x+ay*n.y;
        if(gap(i)<2e-9&&Math.abs(vn)<1e-8&&an<0){ax-=an*n.x;ay-=an*n.y;changed=true;}
      }
      if(!changed)break;
    }
  }
  function drift(dt){s.x+=s.vx*dt+.5*ax*dt*dt;s.y+=s.vy*dt+.5*ay*dt*dt;s.vx+=ax*dt;s.vy+=ay*dt;s.theta+=cfg.omega*dt;s.t+=dt;}
  function impact(i,settle=false){
    const n=normal(i),qx=s.x-radius*n.x,qy=s.y-radius*n.y;
    const wx=-cfg.omega*qy,wy=cfg.omega*qx;
    const incoming=(s.vx-wx)*n.x+(s.vy-wy)*n.y;
    if(incoming>=-1e-11)return false;
    const effectiveE=settle?0:cfg.e;
    const j=-(1+effectiveE)*incoming;
    s.vx+=j*n.x;s.vy+=j*n.y;
    s.wallWork+=j*(wx*n.x+wy*n.y);
    s.loss+=.5*(1-effectiveE*effectiveE)*incoming*incoming;
    const outgoing=(s.vx-wx)*n.x+(s.vy-wy)*n.y;
    s.collisions++;
    s.last={i,qx,qy,wx,wy,nx:n.x,ny:n.y,incoming,outgoing,e:effectiveE,t:s.t};
    return true;
  }
  function project(){
    // Alternating projection handles both edge constraints near a vertex.
    for(let pass=0;pass<32;pass++){
      let worst=0;
      for(let i=0;i<6;i++){
        const d=gap(i),penetration=Math.max(0,-d);
        s.peakPen=Math.max(s.peakPen,penetration);worst=Math.max(worst,penetration);
        if(penetration>0){const n=normal(i);s.x+=(penetration+EPS)*n.x;s.y+=(penetration+EPS)*n.y;}
      }
      if(worst<1e-11)break;
    }
  }
  function advance(dt){
    if(sleeping){s.t+=dt;return;}
    let left=dt;
    while(left>1e-12){
      support();
      // Bound travel and angular change as well as elapsed time.
      const speed=Math.hypot(s.vx,s.vy)+Math.abs(cfg.omega)*(R+radius)+cfg.g*cfg.step;
      const h=Math.min(left,cfg.step,.025*radius/Math.max(speed,1e-6),.004/Math.max(Math.abs(cfg.omega),1e-6));
      let remain=h,events=0;
      while(remain>1e-12&&events<16){
        let hit=-1,hitT=remain+1;
        for(let i=0;i<6;i++){
          const start=gap(i),end=gap(i,remain);
          if(end>=-EPS)continue;
          let lo=0,hi=remain;
          if(start>EPS){
            for(let k=0;k<36;k++){const mid=(lo+hi)/2;if(gap(i,mid)>0)lo=mid;else hi=mid;}
          }else{
            const n=normal(i),rel=(s.vx+cfg.omega*s.y)*n.x+(s.vy-cfg.omega*s.x)*n.y;
            if(rel<-1e-8)hi=0;
            else {
              // Contact at rest: support forces are resolved at the substep end.
              continue;
            }
          }
          if(hi<hitT){hitT=hi;hit=i;}
        }
        if(hit<0){drift(remain);remain=0;break;}
        drift(hitT);remain-=hitT;
        impact(hit);project();support();events++;
      }
      if(remain>1e-12){s.fallbacks++;drift(remain);}
      project();
      // Low-speed persistent contact uses an inelastic support impulse.
      for(let pass=0;pass<4;pass++){
        let any=false;
        for(let i=0;i<6;i++)if(gap(i)<2e-9)any=impact(i,true)||any;
        if(!any)break;
      }
      left-=h;
      // Suppress the Zeno sequence at a stable, stationary bottom corner.
      if(cfg.omega===0&&cfg.g>0&&cfg.e<1&&Math.hypot(s.vx,s.vy)<.03){
        const rv=(A-radius)/A;
        const vv=Array.from({length:6},(_,i)=>({x:rv*Math.cos(s.theta+i*Math.PI/3),y:rv*Math.sin(s.theta+i*Math.PI/3)}));
        const bottom=vv.reduce((a,b)=>a.y<b.y?a:b);
        if(Math.hypot(s.x-bottom.x,s.y-bottom.y)<8e-5){
          s.loss+=.5*(s.vx*s.vx+s.vy*s.vy);s.vx=0;s.vy=0;
          s.x=bottom.x*(1-1e-10);s.y=bottom.y*(1-1e-10);
          sleeping=true;s.t+=left;left=0;
        }
      }
    }
  }
  function diagnostics(){return {energy:energy(),error:energy()-s.E0-s.wallWork+s.loss,minGap:Math.min(...Array.from({length:6},(_,i)=>gap(i))),...s};}
  function setState(values){sleeping=false;Object.assign(s,values);s.E0=energy();s.wallWork=0;s.loss=0;s.peakPen=0;s.collisions=0;s.last=null;s.fallbacks=0;}
  return {s,cfg,advance,diagnostics,setState,normal,gap,energy,R,radius};
}
global.createHexSimulation=createSimulation;
})(window);
