from pathlib import Path
import base64, io, json
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/week01_3/W1_3_Keras_Optimizers.ipynb"
import os
NOTEBOOK = Path(os.environ.get("OPTIMIZER_NOTEBOOK", str(NOTEBOOK)))
OUTPUT = Path(os.environ.get("OPTIMIZER_PDF", str(ROOT / "lectures/week01_3_keras_optimizers.pdf")))
FONTS = Path(r"C:\Windows\Fonts")
for name, file in (("TimesNR","times.ttf"),("TimesNR-Bold","timesbd.ttf"),
                   ("TimesNR-Italic","timesi.ttf"),("TimesNR-BoldItalic","timesbi.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONTS/file)))
pdfmetrics.registerFontFamily("TimesNR", normal="TimesNR",bold="TimesNR-Bold",
                              italic="TimesNR-Italic",boldItalic="TimesNR-BoldItalic")
W, H = letter
M, R = 53, W-53
BODY = R-M
c = canvas.Canvas(str(OUTPUT), pagesize=letter)
c.setTitle("Week 1.3 - Optimizers for Scientific Machine Learning with TensorFlow and Keras")
c.setAuthor("Ehsan Roohi")
c.setSubject("FlowMLLab student lecture and CPU laboratory")
styles = {
    "body": ParagraphStyle("body",fontName="TimesNR",fontSize=11.7,leading=16.5),
    "small": ParagraphStyle("small",fontName="TimesNR",fontSize=10.5,leading=14),
    "caption": ParagraphStyle("caption",fontName="TimesNR",fontSize=10.6,leading=14.2),
    "title": ParagraphStyle("title",fontName="TimesNR-Bold",fontSize=27,leading=30),
    "sub": ParagraphStyle("sub",fontName="TimesNR-Bold",fontSize=14,leading=18),
    "eq": ParagraphStyle("eq",fontName="TimesNR-Italic",fontSize=13,leading=18,alignment=TA_CENTER),
    "cell": ParagraphStyle("cell",fontName="TimesNR",fontSize=10.8,leading=13.4),
}
page_num=0
y=0

def draw_p(text,x,top,width,style):
    obj=Paragraph(text,styles[style])
    _,height=obj.wrap(width,H)
    obj.drawOn(c,x,top-height)
    return height

def p(text,style="body",gap=9):
    global y
    y-=draw_p(text,M,y,BODY,style)+gap

def sub(text):
    p(text,"sub",4)

def eq(text):
    p(text,"eq",7)

def rule(yy):
    c.setStrokeGray(0.4)
    c.setLineWidth(0.45)
    c.line(M,yy,R,yy)

def page(title, tag):
    global page_num,y
    if page_num:
        assert y>=56,(page_num,y)
        c.showPage()
    page_num+=1
    c.setFillGray(0)
    c.setFont("TimesNR",10)
    c.drawString(M,H-34,"FlowMLLab  |  Week 1.3")
    c.drawRightString(R,H-34,"TensorFlow and Keras")
    rule(H-43)
    rule(45)
    c.drawString(M,29,"Scientific machine learning  |  Ehsan Roohi")
    c.drawRightString(R,29,str(page_num))
    anchor=f"p{page_num}"
    c.bookmarkPage(anchor)
    c.addOutlineEntry(title,anchor,level=0,closed=False)
    y=H-66
    p(tag.upper(),"small",7)
    p(title,"title",13)

def note(title,text):
    global y
    obj=Paragraph(f"<b>{title}</b> {text}",styles["small"])
    _,hh=obj.wrap(BODY-24,H)
    c.setStrokeGray(0.68)
    c.setFillGray(0.96)
    c.roundRect(M,y-hh-18,BODY,hh+18,4,stroke=1,fill=1)
    c.setFillGray(0)
    obj.drawOn(c,M+12,y-hh-8)
    y-=hh+28

def table(headers,rows,widths):
    global y
    def cell(s):
        return Paragraph(s,styles["cell"])
    t=Table([[cell("<b>"+h+"</b>") for h in headers]]+[[cell(v) for v in row] for row in rows],
            colWidths=widths,hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN",(0,0),(-1,-1),"TOP"),("BACKGROUND",(0,0),(-1,0),colors.HexColor("#ededed")),
        ("LINEABOVE",(0,0),(-1,0),0.6,colors.grey),("LINEBELOW",(0,0),(-1,0),0.5,colors.grey),
        ("LINEBELOW",(0,1),(-1,-1),0.3,colors.HexColor("#bbbbbb")),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    _,hh=t.wrap(BODY,H)
    t.drawOn(c,M,y-hh)
    y-=hh+12

def image(idx,cap,maxh=375):
    global y
    im=ImageReader(io.BytesIO(figures[idx]))
    ww,hh=im.getSize()
    s=min(BODY/ww,maxh/hh)
    c.drawImage(im,M+(BODY-ww*s)/2,y-hh*s,ww*s,hh*s)
    y-=hh*s+9
    p(cap,"caption",10)

def schematic(labels):
    global y
    space=18
    box=(BODY-space*(len(labels)-1))/len(labels)
    for i,label in enumerate(labels):
        x=M+i*(box+space)
        c.setFillGray(0.96)
        c.setStrokeGray(0.55)
        c.roundRect(x,y-49,box,47,4,stroke=1,fill=1)
        c.setFillGray(0)
        draw_p(label,x+7,y-10,box-14,"small")
        if i<len(labels)-1:
            a=x+box
            c.line(a+2,y-25,a+space-3,y-25)
            c.line(a+space-8,y-22,a+space-3,y-25)
            c.line(a+space-8,y-28,a+space-3,y-25)
    y-=62

def link(url,label):
    return f'<link href="{escape(url)}"><u>{escape(label)}</u></link>'

nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
figures=[base64.b64decode("".join(out["data"]["image/png"]))
         for cell in nb["cells"] for out in cell.get("outputs",[])
         if "image/png" in out.get("data",{})]
assert len(figures)==8


page("Optimizers for Scientific Machine Learning", "Student lecture and CPU laboratory")
p("TensorFlow and Keras  |  Companion to Week 1.3","sub")
p("Ehsan Roohi · FlowMLLab")
p("A neural network can miss a narrow peak even when its loss decreases. Before changing the network, ask which part of the experiment is limiting it: the update rule, the scale of the target, the distribution of observations, or the model's ability to represent the feature. This lab gives you a controlled way to distinguish those mechanisms.")
note("Prerequisites.", "Complete Week 1, Notebook 02, especially the sine-fit exercise. You should know what a gradient is and be able to distinguish training, validation and test data. The lab uses generated analytic data and runs on a CPU.")
sub("What you will learn")
p("1. Derive the gradient of mean squared error for one parameter and predict when a fixed step overshoots.")
p("2. Explain momentum, Adam, and the curvature information used by L-BFGS.")
p("3. Compare full-batch and staged routes with matched initialization, fixed data and an explicit evaluation budget.")
p("4. Choose an optimizer using validation evidence, then report held-out error both globally and where the feature matters.")
sub("Reading route")
p("Pages 2–5 build the update rules. Pages 6–16 walk through the controlled experiments and connect them to physics-informed neural networks (PINNs). Pages 17–18 contain the assignment and references.")
p(link("https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/notebooks/week01/02_tensorflow_for_ai_fluids.ipynb","Week 1 Notebook 02") + " · " + link("https://github.com/Ehsan-Roohi/FlowMLLab/blob/main/lectures/week01_02_tensorflow_keras.pdf","TensorFlow/Keras prerequisite guide"),"small")
note("Plan your time.", "Allow 45–60 minutes for the core lesson and 30–45 minutes for the extensions. Use a CPU runtime. Compare machine timings only after repeating on the same machine and under the same monitoring settings.")

page("What an optimizer changes", "01  |  From a prediction to an update")
p("A dense network represents a function ŷ = f<sub>θ</sub>(x), where θ contains weights and biases. The activation lets a stack of dense layers represent a nonlinear function. TensorFlow records the computations, then automatic differentiation returns the partial derivative of the loss with respect to every trainable parameter.")
schematic(["Training points<br/>xᵢ and targets yᵢ", "Forward pass<br/>Predict ŷᵢ", "Loss and gradient<br/>J(θ) and grad J(θ)", "Optimizer<br/>Update θ"])
eq("J(θ) = (1/N) Σᵢ [f<sub>θ</sub>(xᵢ) − yᵢ]²")
p("For a single weight and prediction ŷᵢ = wxᵢ, the chain rule gives the derivative below. The error term gives the mismatch and xᵢ gives the sensitivity of the prediction to w.")
eq("dJ/dw = (2/N) Σᵢ (wxᵢ − yᵢ)xᵢ")
sub("Worked example: a step can help or hurt")
p("Use one point x = 2 with target y = 3 and start at w = 0.5. The prediction is 1, the loss is 4, and the gradient is −8. A learning rate of 0.1 gives w = 1.3 and loss 0.16. A learning rate of 0.3 gives w = 2.9 and loss 7.84, after overshooting the minimizing weight 1.5.")
eq("w<sub>new</sub> = w − η (dJ/dw)")
p("Here the error in the weight is multiplied by 1 − 8η at each gradient step. Its magnitude shrinks when 0 &lt; η &lt; 0.25. Between 0.125 and 0.25 the sign alternates while the error shrinks. A network has many directions of curvature and there is usually no single stability threshold you can obtain from inspection.")
note("Self-check.", "At η = 0.125, the example reaches w = 1.5 in one step. Check why changing x changes the stability range in this same example.")

page("Momentum and Adam", "02  |  First-order updates")
p("Let θ<sub>k</sub> be the weights and biases before update k and g<sub>k</sub> = grad J(θ<sub>k</sub>) the gradient of the full-training-set mean squared error J.")
sub("Gradient descent with momentum")
eq("v<sub>k+1</sub> = μv<sub>k</sub> − ηg<sub>k</sub>,    θ<sub>k+1</sub> = θ<sub>k</sub> + v<sub>k+1</sub>")
p("The velocity v starts at zero. The learning rate η is 0.015 and μ = 0.85 sets how much of the previous update carries forward. When the gradients point in similar directions, momentum builds speed. When they reverse, the accumulated velocity can cause overshoot.")
sub("Adam")
eq("m<sub>k+1</sub> = β₁m<sub>k</sub> + (1 − β₁)g<sub>k</sub>")
eq("q<sub>k+1</sub> = β₂q<sub>k</sub> + (1 − β₂)g<sub>k</sub>²")
eq("c<sub>k+1</sub> = √(1 − β₂<super>k+1</super>)/(1 − β₁<super>k+1</super>)")
eq("θ<sub>k+1</sub> = θ<sub>k</sub> − η c<sub>k+1</sub> m<sub>k+1</sub>/(√(q<sub>k+1</sub>) + ε)")
p("The moving mean m and mean square q both start at zero. Their decay rates are β₁ = 0.9 and β₂ = 0.999. The factor c corrects for zero initialization. The Keras update uses ε = 10<super>−7</super> to stabilize division, and the default learning rate in this lab is η = 0.01. Squaring, square roots and division of vectors act separately on each parameter.")
note("What to look for.", "Momentum smooths the direction of travel. Adam also changes the effective step scale for each parameter. Compare the two runs with the same data and initial weights.")
p("See " + link("https://keras.io/api/optimizers/sgd/","Keras SGD") + " and " + link("https://keras.io/api/optimizers/adam/","Keras Adam") + ".","small")

page("Newton, BFGS and limited memory", "03  |  Curvature and line search")
p("The gradient describes slope. The Hessian H contains second derivatives and describes local curvature. For a smooth objective near the current point, the quadratic approximation uses both.")
eq("J(θ + d) ≈ J(θ) + gᵀd + ½dᵀHd")
p("When H is positive definite, minimizing this local quadratic gives Hd = −g. A direction of large curvature tends to receive a shorter step. At a neural network's scale, forming and solving with the full Hessian is expensive. Away from a minimum, a Newton direction also may fail to be downhill.")
sub("What L-BFGS retains")
p("Broyden–Fletcher–Goldfarb–Shanno (BFGS) updates an approximation to the inverse Hessian using the change in parameters s and the change in gradient y. Limited-memory BFGS (L-BFGS) retains only recent pairs and uses them to obtain the direction without storing a full matrix.")
eq("s<sub>k</sub> = θ<sub>k+1</sub> − θ<sub>k</sub>,    y<sub>k</sub> = g<sub>k+1</sub> − g<sub>k</sub>")
eq("d<sub>k</sub> = −B<sub>k</sub>g<sub>k</sub>,    θ<sub>k+1</sub> = θ<sub>k</sub> + α<sub>k</sub>d<sub>k</sub>")
p("B<sub>k</sub> approximates the inverse Hessian, d<sub>k</sub> is the search direction, and α<sub>k</sub> is the step length. The BFGS update underlying L-BFGS uses the identity matrix I, ρ = 1/(yᵀs) and V = I − ρsyᵀ. The superscript T denotes transpose. All gradient differences use the same training set.")
eq("B<sub>k+1</sub> = VB<sub>k</sub>Vᵀ + ρssᵀ")
sub("Trials, accepted steps and stopping")
p("A line search tries θ + αd, evaluates the loss and gradient, and can reject α before accepting a step. A rejected trial still consumes a counted joint evaluation. The notebook uses SciPy's L-BFGS-B interface with no box bounds and passes the TensorFlow gradient. When the budget runs out during a line search, it restores the last accepted state.")
note("Worked accounting.", "The 96 extra calls in a run with 600 joint evaluations and 504 accepted updates include search setup and trials.")

page("What the hybrid tests", "04  |  A fair comparison")
schematic(["Same initial<br/>weights and data", "270 Adam updates<br/>in the hybrid", "L-BFGS to the<br/>600-call limit", "Fixed validation<br/>and final test"])
p("The hybrid spends the first part of its budget on Adam, then hands the same weights to the curvature method. A staged method can work well when the first phase finds a useful region and the second reduces residual error. It also can waste budget if the second phase could have made faster progress from the start.")
table(["Comparison","Question answered"],[
    ["Hybrid versus Adam for the full budget","Does switching help compared with continuing from the same Adam prefix?"],
    ["Hybrid versus L-BFGS from the start","Does the Adam prefix earn its cost?"],
    ["Momentum versus Adam","How do these fixed learning-rate settings behave under the same full-batch objective?"],
],[202,306])
p("The notebook checks that all methods start at the same weights and that Adam and the hybrid agree at their common recorded points before the switch. Count joint training loss/gradient calls for algorithmic work. Forward-only diagnostics and tracing are outside that budget. Report time separately: these implementations use different surrounding code and monitoring costs.")
sub("Keep the comparisons straight")
p("A method can have the lowest training loss and a worse validation score. Use validation RMSE in original units to choose candidate settings. Compare elapsed seconds on the same machine, ideally with repeated runs if you want to rank speed. When changing optimizers, keep the sample coordinates, architecture, precision, and initialization fixed.")
note("Check before interpreting.", "If the two fixed-set loss curves disagree before the switch for Adam and the hybrid, verify that you restarted from the same weights and used the same learning rate and data.")

page("Data roles and target scaling", "05  |  Protect the comparison")
table(["Role","Use in the notebook","Default points"],[
  ["Training","Fit weights and the target scaling constants.","121"],
  ["Validation","Compare architecture, optimizer and sampling choices.","61"],
  ["Test","Report errors after recording a choice.","601"],
],[100,298,110])
p("Training includes the endpoints. Validation and test use different shifted coordinates inside the same interval. This is a controlled interpolation study. A shared analytic target makes ground truth available, but the test scores are hidden by default to keep the model selection procedure clear.")
eq("z = (y − μ<sub>train</sub>) / σ<sub>train</sub>,    ŷ = μ<sub>train</sub> + σ<sub>train</sub>ẑ")
p("Use only training targets to calculate μ and σ. Apply those same values to validation targets. Transform the predictions back before evaluating in the original units. If the target is nearly constant on the selected training points, a standard deviation close to zero is a warning to inspect the sampling.")
eq("RMSE = √[(1/N) Σᵢ (ŷᵢ − yᵢ)²]")
p("For a fixed split and one model, original-unit RMSE equals σ<sub>train</sub> times the square root of the normalized MSE. When you change the training count and refit σ, or toggle normalization, compare original-unit errors. Different normalized loss scales make a direct comparison misleading.")
note("Worked example.", "For μ = 1000 and σ = 25, a model output of 0.4 returns ŷ = 1010. If the true value is 1012, the signed error is −2 in the original units. A normalized output of 0.4 is not itself a prediction in those units.")

page("How hard are the targets?", "06  |  Choosing a useful test problem")
image(0,"<b>Figure 1.</b> Four analytic targets from the notebook. The interval [0.65, 0.79] is used for separate diagnostics around the localized feature.",276)
table(["Target","What it can reveal"],[
    ["sin(2πx)","A smooth control. Even here the schedules separate at a finite work budget."],
    ["x² sin(2πx)","An amplitude-modulated bridge from Notebook 02. Compare against the plain sine before changing the architecture."],
    ["0.65 sin(2πx) + 0.20 sin(8πx) + 0.08 cos(16πx)","Small-scale waves can be missed even when the overall trend looks right. Check the error profile."],
    ["0.20 sin(2πx) + exp[−((x − 0.72)/0.035)²]","A narrow feature can be poorly sampled or poorly fitted. Report error inside the window and across the domain."],
],[210,298])
p("All four targets have a complete convergence and error comparison. A fifth case adds a chirp from 2 to 22 cycles per unit interval and tests the staged optimizer used in the deep-cavity study.","small")



page("What the CPU comparisons show", "07  |  Validation in original units")
names = ["Sine", "Amplitude-modulated", "Multiscale", "Localized"]
routes = ["Momentum", "Adam", "L-BFGS", "Adam → L-BFGS"]
def validation(case, method, offset=0):
    cell = next(cell for cell in nb["cells"] if cell["cell_type"] == "code" and
                cell["source"] and "".join(cell["source"]).startswith("CASES =" if case in names else 'TARGETS["Chirp"]'))
    for output in cell["outputs"]:
        block = "".join(output.get("data", {}).get("text/markdown", []))
        if block.startswith(f"**{case}**"):
            row = next(row for row in block.splitlines() if row.startswith(f"| {method} |"))
            return float(row.split("|")[5 + offset].strip())
    raise ValueError((case, method))

table(["Route","Sine","Amplitude-modulated","Multiscale","Localized"],[
    [route]+[f"{validation(case, route):.4g}" for case in names] for route in routes
],[124,74,110,99,101])
p("Root-mean-square errors (RMSE) are reported on the fixed validation grid and in original target units. The settings are a two-hidden-layer tanh network of width 24, a limit of 600 joint training loss-and-gradient calls, and seed 2026. The hybrid switches at call 270. Use the plots for comparisons at intermediate work and elapsed time.")
for case in names:
    winner = min(routes, key=lambda route: validation(case, route))
    p(f"<b>{case}.</b> {winner} gives the lowest recorded validation RMSE ({validation(case,winner):.4g}) at this budget.", "small", 3)
sub("What makes an optimizer effective here?")
p("Different loss directions respond to different step sizes. A method that uses useful curvature estimates can take smaller steps across a steep direction while moving farther along a shallow one. On a fixed objective, a line search can check whether its proposed step actually lowers the loss sufficiently. Adam adapts to the recent magnitudes of different gradient components. SOAP applies the adaptation in a rotated basis.")
p("A good fit must resolve the regions you care about at acceptable cost. A method that reduces training loss quickly can still miss a narrow peak or a rapidly oscillating section. Compare the signed error with the reference and check whether additional samples or a more suitable representation reduce the remaining error.")
note("Limit of the evidence.", "These results compare specified settings and one paired initialization. A ranking is a property of this experiment. Vary the seed and measure the relevant region before recommending a route for another problem.")

examples = [
    ("Sine", "Smooth control", 1, "Compare early and late progress. Is the route with the lowest final validation error also the cheapest route to a tolerance you choose? Mark the first crossing and report both joint calls and seconds."),
    ("Amplitude-modulated", "Uneven amplitude", 2, "The amplitude approaches zero near the left boundary and grows toward the right. Compare signed errors on each half. Use a common original-unit metric so the quieter part cannot be mistaken for an easier optimizer solely because of its smaller target."),
    ("Multiscale", "Coarse and fine oscillations", 3, "Follow the fast ripples on the reference and check where all predicted curves miss them. If changing the optimizer barely changes the missed ripples, test a representation or width change with the original split and the same seed."),
    ("Localized", "A narrow feature", 4, "The shaded interval contains the peak. Training sites are marked in the lower-right panel. Compare the signed error and RMSE in this interval with the whole-domain value. Determine which route misses the height and whether the error changes sign on the shoulders."),
]
for number, (case, subtitle, idx, discussion) in enumerate(examples, start=8):
    page(case, f"{number:02d}  |  {subtitle}")
    image(idx, f"<b>Figure {idx+1}.</b> {case}. Top: training MSE by joint calls and validation MSE by elapsed time. Bottom: reference and validation predictions, then signed validation errors. The vertical dotted mark identifies the Adam-to-L-BFGS transition.", 400)
    p(discussion)
    note("Report both measures.", "Give the original-unit validation RMSE and describe a specific location where the profile exposes an error. Use the elapsed-time axis cautiously when the differences are small.")

page("Diagnose a jump in loss", "12  |  Is it the update or the measurement?")
image(5,"<b>Figure 6.</b> Two Adam learning rates with identical initialization and fixed training and validation grids. Each displayed point is evaluated after the same accepted update on the same diagnostic grid.",232)
p("Read the reported maximum consecutive-loss ratios and locate the jumps on the curves. Both fixed-set curves respond to the large early step, so a change in batch membership cannot explain that jump.")
table(["Observation","Next check"],[
    ["Fixed training and validation losses spike together","Rerun with the same seed and data at a smaller learning rate."],
    ["Only the online training log jumps","Measure again on the same fixed training grid and check batch order."],
    ["Training improves while validation drifts","Check coverage, compare capacity and repeat with a paired seed."],
    ["A jump occurs near a stage switch","Check whether the objective, points, learning rate or logged quantity changed."],
],[232,276])
note("Record the evidence.", "Report the update number, both loss values before and after it, and what changed in the rerun. An isolated increase is evidence of an uphill step on the measured objective rather than a complete test of convergence.")

page("When scaling changes the fit", "13  |  Same function, same interval")
image(6,"<b>Figure 7.</b> Controlled scaling example with an analytic target near 1,000. The same Adam setup and 220 evaluation calls are used each time. Both panels use the original units.",236)
p("Compare the reported original-scale validation errors. Train-only target scaling removes the large offset from the quantity the network needs to learn. The final output transformation restores the original level.")
p("The normalized run still has visible deviations. Interpret its prediction and error profiles in addition to the headline RMSE. An output rescaling affects the regression parameterization and gradient magnitudes. The experiment says the specified scaling helped under the stated time and model budget.")
sub("Repeat as a controlled experiment")
p("Keep the seed, optimizer, learning rate, grid and work budget fixed. Confirm that raw and normalized models are each evaluated against the same original target. Test whether a longer raw-target run closes the gap before claiming that scaling changes the best attainable fit.")
note("Common mistake.", "If you transform the predictions but compare them to normalized targets, the reported RMSE has no interpretable units. Compute error after predictions and reference have been placed on the same scale.")

page("From supervised fitting to PINNs", "14  |  Derivatives, fixed residuals and flow evidence")
p("A physics-informed neural network can minimize a residual rather than fit field labels. Two networks with similar value error can have very different derivative error. For example, a small added wave with amplitude a at wavenumber k has first derivative amplitude ak and second derivative amplitude ak².")
eq("e(x) = a sin(kx),    e′(x) = ak cos(kx),    e″(x) = −ak² sin(kx)")
p("In the sine case, tape-differentiate the model output after undoing scaling. Compare it against 2π cos(2πx). A piecewise linear ReLU model has zero second derivative away from its breakpoints. That can be a poor match for a second-derivative residual even when its values look good.")
schematic(["Training residual<br/>or data fit", "Fixed diagnostic<br/>on chosen points", "Independent field<br/>comparison", "Inspect local<br/>flow features"])
p("The deep cavity workflow by Roohi and McDevitt used SOAP with periodically redrawn regional samples followed by fixed-point self-scaled Broyden. In the kinetic-shock example, the Adam and self-scaled Broyden stages could also use different evaluation supports. Raw logs from those phases must identify what was measured. The notebook also runs the SOAP–SSB sequence on a fixed regression objective and marks each phase transition.")
p("For cavity or shock studies, supplement residual loss with independent physics checks. Examples include regional velocity and vorticity discrepancies for a cavity, or profiles through a shock. Evaluate derivative quality and local behavior as well as a domain-wide score.")
note("Read next.", link("https://arxiv.org/html/2610.38814v2#S3.SS4.SSS1","Deep cavity optimization method") + " · " + link("https://github.com/Ehsan-Roohi/FlowMLLab/tree/main/weeks/week13","FlowMLLab Week 13 cavity PINNs"))


page("The deep-cavity optimizer sequence", "15  |  SOAP and self-scaled Broyden")
p("The deep-cavity study by Roohi and McDevitt used an Adam-like Shampoo-based optimizer (SOAP) on periodically refreshed regional samples. The final stage applied self-scaled Broyden (SSB) to a fixed residual objective. The CPU lab isolates the same sequence with one chirp and a small network.")
eq("G̃ = Q<sub>L</sub>ᵀ G Q<sub>R</sub>,    W<sub>new</sub> = W − η Q<sub>L</sub> Ũ Q<sub>R</sub>ᵀ")
p("W is a dense weight matrix and G its loss gradient. Q<sub>L</sub> and Q<sub>R</sub> are orthogonal bases estimated from moving averages of GGᵀ and GᵀG. Ũ is the Adam-style direction in these coordinates, and η is the learning rate. Bias gradients use Adam-style moments without rotation.")
eq("B<sub>new</sub> = [B − zzᵀ/b + φbuuᵀ]/τ + ssᵀ/a")
p("In SSB, B is the inverse-Hessian approximation, s is the accepted displacement, and y is the gradient difference. Set z = By, a = yᵀs, b = yᵀz, and u = s/a − z/b. The self-scale τ and the Broyden coefficient φ depend on the accepted step and curvature. The notebook reports any BFGS fallback.","small")
image(7,"<b>Figure 8.</b> Increasing-frequency chirp. The curve frequency goes from 2 to 22 cycles per unit interval. Each hybrid switches at call 240. Time-axis switch markers follow each route’s color.",273)
note("Matched design.", "The network widths are 16 and 16 and the full budget is 600 joint calls. Initialization and training points are shared. The research solver uses a different objective, sampling schedule and curvature storage.")

page("Student assignment", "16  |  Experiments and what to hand in")
table(["Exercise","Change or diagnostic","Evidence to submit"],[
  ["1. Point counts","Use 31, 61 and 121 training points for Localized. Keep validation fixed. Then vary validation count separately.","Peak-window and global validation RMSE, a signed error profile, and the number of points inside the window."],
  ["2. Network capacity","Compare (12,), (24,24) and (32,32,32). Keep the split, optimizer and budget fixed.","Parameters, joint calls, seconds and error. Say whether time or approximation quality set the limit."],
  ["3. Activation","Compare tanh, ReLU and swish on Sine. For Sine compare the differentiated prediction with the known derivative.","Original-unit value and derivative RMSE. Explain which activation you would try for a second-derivative residual."],
  ["4. Loss jump and scaling","Reproduce the two learning rates. Repeat with the large-offset target and toggle train-only scaling.","Before-and-after losses at the largest jump and original-unit errors for both target representations."],
  ["5. Switch point","Test 120, 270 and 420 joint calls before the L-BFGS phase. Include Adam-only and L-BFGS-only controls.","Choose using validation. Lock the settings and then report test error once."],
  ["6. Cavity sequence","On Chirp compare Adam → L-BFGS, Adam → SSB, SOAP → SSB and SOAP alone. Repeat for seeds 2027 and 2028.","Global and x ≥ 0.75 validation RMSE, time, genuine SSB updates and fallbacks. Describe the limiting failure mode."],
],[104,190,214])
note("Submission.", "Include a settings table, two annotated plots, an error-and-cost table and a short conclusion. For every recommendation name the objective, the comparison budget, and the metric. Exercises 1–5 use the main notebook controls. Exercise 6 is the research extension.")

page("Reading the evidence", "17  |  Interpretation and references")
methods = ["Adam","Adam → L-BFGS","SOAP","Adam → SSB","SOAP → SSB"]
table(["Chirp route","Whole interval RMSE","Final quarter RMSE"], [
    [m, f"{validation('Chirp', m):.4g}", f"{validation('Chirp', m, offset=1):.4g}"] for m in methods
], [238,135,135])
winner=min(methods,key=lambda m:validation("Chirp",m))
p(f"<b>Saved result.</b> {winner} has the lowest validation RMSE for this seed and budget. Compare it with the actual profile in Figure 8. All five routes give about 0.73 RMSE in the final quarter. The gain over the full interval mainly comes from the slower part of the signal. First try a controlled change in width or input features if the signed error oscillates around zero over several unresolved cycles.")
p("<b>Comparison with published exercises.</b> PyTorch’s tutorial fits sine with a cubic polynomial and an optimizer. SciPy uses the curved Rosenbrock valley to explain numerical minimizers. Tancik and co-authors show why ordinary multilayer networks can struggle to learn high-frequency signals in low-dimensional coordinates. Kiyani and co-authors compare optimizers directly on several PINN residual problems. This lab puts function fitting between the introductory examples and full physical residual problems.","small")
sub("Sources")
references = [
    ("https://docs.pytorch.org/tutorials/beginner/examples_nn/polynomial_optim.html", "PyTorch: fitting sine with a polynomial and an optimizer"),
    ("https://docs.scipy.org/doc/scipy/tutorial/optimize.html", "SciPy: the Rosenbrock valley and optimization methods"),
    ("https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html", "Tancik et al. (2020): Fourier features and high-frequency regression"),
    ("https://arxiv.org/html/2610.38814v2#S3.SS4.SSS1", "Roohi and McDevitt: deep-cavity optimizer sequence"),
    ("https://arxiv.org/abs/2409.11321", "Vyas et al.: SOAP"),
    ("https://arxiv.org/abs/2405.04230", "Urbán, Stefanou and Pons: self-scaled Broyden"),
    ("https://arxiv.org/abs/2501.16371", "Kiyani et al.: optimizer benchmarks for PINNs"),
]
for url, label in references:
    p(link(url,label),"small",3)
assert y>=56,(page_num,y)
c.save()
