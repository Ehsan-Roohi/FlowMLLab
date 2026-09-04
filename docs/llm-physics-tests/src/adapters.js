(function(global){
  const ids=['extra-high','medium','light'];
  function create(id,{omega=.7,g=9.81,e=.85}={}){
    let solver,state;
    if(id==='extra-high'){solver=global.createHexSimulation({omega,g,e});state=solver.s;}
    else if(id==='medium'){solver=global.makeHexMedium({w:omega,g,e});state=solver.s;}
    else {solver=global.hexLightPhysics({omega,gravity:g,restitution:e});state=solver.state;}
    return {id,solver,state,advance:dt=>solver.advance(dt),read:()=>({x:state.x,y:state.y,vx:state.vx,vy:state.vy,t:state.t??state.time,theta:.12+omega*(state.t??state.time)}),set(values){
      if(id==='extra-high')solver.setState({...values,t:0,theta:.12});
      else Object.assign(state,values,{[id==='light'?'time':'t']:0});
    }};
  }
  global.HexAdapters={ids,create};
})(typeof window==='undefined'?globalThis:window);
