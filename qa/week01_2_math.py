"""Vector mathematical typography for the Week 1.2 lecture (no TeX install)."""
from matplotlib import rc_context, font_manager
from matplotlib.textpath import TextPath
from matplotlib.path import Path
from reportlab.platypus import Flowable


class Equation(Flowable):
    """Draw MathText outlines directly into PDF, including fraction rules."""
    def __init__(self, expression, max_width, size=13):
        super().__init__()
        for filename in ('times.ttf', 'timesi.ttf', 'timesbd.ttf'):
            font_manager.fontManager.addfont('C:/Windows/Fonts/' + filename)
        with rc_context({'mathtext.fontset': 'custom',
                         'mathtext.rm': 'Times New Roman',
                         'mathtext.it': 'Times New Roman:italic',
                         'mathtext.bf': 'Times New Roman:bold',
                         'mathtext.fallback': 'stix'}):
            self.outline = TextPath((0, 0), '$' + expression + '$', size=size)
        self.bounds = self.outline.get_extents()
        self.scale = min(1., max_width / self.bounds.width)
        self.width = max_width
        self.height = self.bounds.height * self.scale + 13

    def draw(self):
        c = self.canv
        c.saveState()
        c.translate((self.width-self.bounds.width*self.scale)/2,
                    6-self.bounds.ymin*self.scale)
        c.scale(self.scale, self.scale)
        c.translate(-self.bounds.xmin, 0)
        c.setFillColorRGB(0, 0, 0)
        path = c.beginPath()
        previous = None
        for vertices, code in self.outline.iter_segments(curves=True):
            if code == Path.MOVETO:
                path.moveTo(*vertices); previous = vertices
            elif code == Path.LINETO:
                path.lineTo(*vertices); previous = vertices
            elif code == Path.CURVE3:
                x0,y0 = previous; x1,y1,x2,y2 = vertices
                path.curveTo(x0+2*(x1-x0)/3, y0+2*(y1-y0)/3,
                             x2+2*(x1-x2)/3, y2+2*(y1-y2)/3, x2,y2)
                previous = vertices[-2:]
            elif code == Path.CURVE4:
                path.curveTo(*vertices); previous = vertices[-2:]
            elif code == Path.CLOSEPOLY:
                path.close()
        c.drawPath(path, stroke=0, fill=1)
        c.restoreState()
