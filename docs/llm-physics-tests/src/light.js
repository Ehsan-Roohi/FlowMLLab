
window.hexLightPhysics=function(parameters={}){
  const config={omega:0.7,gravity:9.81,restitution:0.85,...parameters};
  const radius=0.065,limit=Math.sqrt(3)/2-radius;
  const state={x:0.08,y:0.36,vx:0.72,vy:0.1,time:0,peak:0,error:0,last:null};
  function normals(){return Array.from({length:6},(_,i)=>{const a=0.12+config.omega*state.time+(i+.5)*Math.PI/3;return [Math.cos(a),Math.sin(a)];});}
  function collide(nx,ny){
    const qx=state.x+radius*nx,qy=state.y+radius*ny;
    const wallX=-config.omega*qy,wallY=config.omega*qx;
    const incoming=(state.vx-wallX)*nx+(state.vy-wallY)*ny;
    if(incoming<=0)return;
    const impulse=(1+config.restitution)*incoming;
    state.vx-=impulse*nx;state.vy-=impulse*ny;
    const outgoing=(state.vx-wallX)*nx+(state.vy-wallY)*ny;
    state.error=Math.max(state.error,Math.abs(outgoing+config.restitution*incoming));
    state.last={incoming,outgoing};
  }
  function penetration(){return Math.max(0,...normals().map(([nx,ny])=>nx*state.x+ny*state.y-limit));}
  function advance(duration){
    while(duration>1e-12){
      const speed=Math.hypot(state.vx,state.vy)+Math.abs(config.omega);
      const dt=Math.min(duration,1/2000,0.0005/Math.max(speed,1));
      state.x+=state.vx*dt;state.y+=state.vy*dt-.5*config.gravity*dt*dt;
      state.vy-=config.gravity*dt;state.time+=dt;duration-=dt;
      const ns=normals();
      for(const [nx,ny] of ns)state.peak=Math.max(state.peak,nx*state.x+ny*state.y-limit);
      for(let pass=0;pass<24;pass++){
        let changed=false;
        for(const [nx,ny] of ns){
          const depth=nx*state.x+ny*state.y-limit;
          if(depth>0){state.x-=(depth+1e-12)*nx;state.y-=(depth+1e-12)*ny;collide(nx,ny);changed=true;}
        }
        if(!changed)break;
      }
    }
  }
  return {state,config,advance,penetration,collide,normals,radius,limit};
};
