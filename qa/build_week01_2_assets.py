"""Scientific grid and coupling-loop diagrams for the Week 1.2 lesson."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textwrap
from matplotlib.patches import Rectangle,FancyBboxPatch
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'lectures/source/week01_2_assets'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['times.ttf','timesbd.ttf','timesi.ttf']:
        p=Path('C:/Windows/Fonts')/name
        if p.exists():font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'mathtext.fontset':'stix','font.size':12,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.set(xlim=(-1.8,2.1),ylim=(-1.65,1.65),aspect='equal');ax.axis('off')
    for x in [-1.5,-.5,.5,1.5]:ax.plot([x,x],[-1.5,1.5],color='.75',lw=.8)
    for y in [-1.5,-.5,.5,1.5]:ax.plot([-1.5,1.5],[y,y],color='.75',lw=.8)
    ax.add_patch(Rectangle((-.5,-.5),1,1,facecolor='#eaf2ff',edgecolor='black',lw=1.5))
    for x,y,t in [(0,0,'P'),(1,0,'E'),(-1,0,'W'),(0,1,'N'),(0,-1,'S')]:
        ax.plot(x,y,'ko',ms=4);ax.text(x+.1,y+.1,t,ha='left',va='bottom',fontweight='bold')
    for x,label in [(.5,'e'),(-.5,'w')]:
        ax.annotate('',(x+.28,0),(x-.05,0),arrowprops={'arrowstyle':'->','color':'#1749aa','lw':2})
        ax.text(x-.04,-.21,'$u_'+label+'$',ha='center',color='#1749aa')
    for y,label in [(.5,'n'),(-.5,'s')]:
        ax.annotate('',(0,y+.28),(0,y-.05),arrowprops={'arrowstyle':'->','color':'#ba2821','lw':2})
        ax.text(-.2,y,'$v_'+label+'$',ha='right',color='#ba2821')
    ax.annotate('',(.5,-1.45),(-.5,-1.45),arrowprops={'arrowstyle':'<->','color':'black'})
    ax.text(0,-1.38,'$h$',ha='center')
    ax.annotate('',(1.95,-.85),(1.55,-.85),arrowprops={'arrowstyle':'->'});ax.text(1.95,-.98,'$x$')
    ax.annotate('',(1.55,-.38),(1.55,-.85),arrowprops={'arrowstyle':'->'});ax.text(1.62,-.38,'$y$')
    for ext in ['pdf','png']:fig.savefig(OUT/f'mac_grid.{ext}',dpi=190,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,7),layout='constrained')
    def box(ax,y,text,height=.11):
        ax.add_patch(FancyBboxPatch((.055,y-height/2),.89,height,boxstyle='round,pad=0.015',facecolor='white',edgecolor='black',lw=.9))
        ax.text(.5,y,textwrap.fill(text,25),ha='center',va='center',fontsize=14)
    def arrow(ax,a,b):ax.annotate('',(.5,b+.06),(.5,a-.06),arrowprops={'arrowstyle':'->','lw':1})
    items=[('SIMPLE',[(.90,'Steady velocity and pressure'),(.73,'Assemble and relax momentum'),(.56,'Momentum predictor'),(.39,'Pressure increment; full velocity correction'),(.22,'Relax pressure; evaluate steady residuals')]),('PISO',[(.90,'Save old-time velocity'),(.73,'Assemble transient momentum; predict'),(.56,'Update H; solve pressure field'),(.39,'Reconstruct velocity'),(.22,'Accept fields; advance time once')]),('PIMPLE',[(.90,'Save old-time velocity'),(.73,'Rebuild momentum; new predictor'),(.56,'Update H; solve pressure field'),(.39,'Reconstruct velocity'),(.22,'Accept fields; advance time once')])]
    for ax,(title,rows) in zip(axes,items):
        ax.set(xlim=(0,1),ylim=(0,1.06));ax.axis('off');ax.set_title(title,fontweight='bold',fontsize=15)
        for y,text in rows:box(ax,y,text)
        for (a,_),(b,_) in zip(rows,rows[1:]):arrow(ax,a,b)
        ax.text(.5,.07,{'SIMPLE':'Repeat steady iterations until both\nmomentum and continuity pass.','PISO':'Inner loop: repeat H / pressure / velocity.\nMomentum coefficients stay fixed.','PIMPLE':'Inner loop: fixed coefficients.\nOuter loop: rebuild coefficients.'}[title],ha='center',va='center',fontsize=10)
        if title=='SIMPLE':
            ax.annotate('',(.02,.73),(.02,.22),arrowprops={'arrowstyle':'->'});ax.plot([.055,.02],[.22,.22],'k-',lw=.8)
        else:
            ax.annotate('',(.02,.56),(.02,.39),arrowprops={'arrowstyle':'->'});ax.plot([.055,.02],[.39,.39],'k-',lw=.8)
            if title=='PIMPLE':
                ax.annotate('',(.98,.73),(.98,.34),arrowprops={'arrowstyle':'->'});ax.plot([.93,.98],[.34,.34],'k-',lw=.8)
    for ext in ['pdf','png']:fig.savefig(OUT/f'loop_map.{ext}',dpi=190,facecolor='white')
    plt.close(fig)
if __name__=='__main__':main()
