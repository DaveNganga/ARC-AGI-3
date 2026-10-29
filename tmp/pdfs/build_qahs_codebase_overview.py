from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase.pdfmetrics import stringWidth

OUT='/tmp/QAHS_Prototype_Codebase_Overview.pdf';W,H=1296,729
NAVY=HexColor('#172A46');INK=HexColor('#213047');MUTED=HexColor('#607083');LINE=HexColor('#CAD4DE')
BLUE=HexColor('#EAF3FA');BLUEB=HexColor('#6E9FC7');TEAL=HexColor('#E7F5F2');TEALB=HexColor('#439B8D');GOLD=HexColor('#FFF4D7');GOLDB=HexColor('#C5962B');LILAC=HexColor('#F2EDF8');LILACB=HexColor('#876DB0');GRAY=HexColor('#F5F7F9');RED=HexColor('#A24C48')
c=canvas.Canvas(OUT,pagesize=(W,H));c.setTitle('QAHS Prototype Codebase Overview');c.setAuthor("Dave Ng'ang'a and Abylay Bissary")
def rr(x,y,w,h,f,s=LINE,r=8,sw=1):c.setFillColor(f);c.setStrokeColor(s);c.setLineWidth(sw);c.roundRect(x,y,w,h,r,fill=1,stroke=1)
def tx(x,y,s,z=9,col=INK,font='Helvetica'):c.setFont(font,z);c.setFillColor(col);c.drawString(x,y,s)
def ctr(x,y,w,s,z=9,col=INK,font='Helvetica-Bold'):c.setFont(font,z);c.setFillColor(col);c.drawCentredString(x+w/2,y,s)
def wrap(s,z,mw,font='Helvetica'):
 out=[];cur=''
 for word in s.split():
  test=(cur+' '+word).strip()
  if stringWidth(test,font,z)<=mw:cur=test
  else:
   if cur:out.append(cur)
   cur=word
 if cur:out.append(cur)
 return out
def para(x,y,s,z=7.6,mw=160,lead=9,col=MUTED,lim=5):
 c.setFont('Helvetica',z);c.setFillColor(col);lines=wrap(s,z,mw)[:lim]
 for i,line in enumerate(lines):c.drawString(x,y-i*lead,line)
 return y-len(lines)*lead
def arrow(x1,y1,x2,y2,col=MUTED,w=1.6):
 import math;c.setStrokeColor(col);c.setLineWidth(w);c.line(x1,y1,x2,y2);a=math.atan2(y2-y1,x2-x1)
 for d in (-.55,.55):c.line(x2,y2,x2-7*math.cos(a+d),y2-7*math.sin(a+d))
def header(subtitle,page):
 c.setFillColor(NAVY);c.rect(0,H-82,W,82,fill=1,stroke=0);tx(42,H-39,'Quantum-Assisted Hypothesis Search',24,white,'Helvetica-Bold');tx(42,H-62,subtitle,11,HexColor('#D8E4EF'));tx(W-374,H-38,"Dave Ng'ang'a and Abylay Bissary",11,white,'Helvetica-Bold');tx(W-374,H-59,f'Advisor: Lanier Watkins  |  {page}',9.5,HexColor('#D8E4EF'))
def fileline(x,y,name,purpose,mw):
 tx(x,y,name,7.8,INK,'Helvetica-Bold');return para(x,y-11,purpose,7.1,mw,8.2,MUTED,2)-4

# PAGE 1
header('Research-readiness extension | 64 hypotheses + compact QUBO + matched conditions','Page 1 of 2')
tx(42,620,'RUNTIME RESEARCH LOOP',10,NAVY,'Helvetica-Bold')
flow=[(42,505,126,'1  Environment','ARC/domain adapter returns state and actions.',BLUE,BLUEB),(187,505,138,'2  Interpret','Deterministic now; OpenAI adapter ready.',TEAL,TEALB),(344,505,144,'3  Hypotheses','Six binary rule dimensions create 64 worlds.',GOLD,GOLDB),(507,505,154,'4  Score','Marginal information gain and progress.',LILAC,LILACB),(680,473,278,'5  Matched search','Same QUBO, observations and action set.',GRAY,LINE),(978,505,128,'6  Execute','Apply selected experiment or solution.',GOLD,GOLDB),(1125,505,129,'7  Update','Record result, update beliefs and repeat.',BLUE,BLUEB)]
for x,y,w,t,b,f,s in flow:
 h=124 if x==680 else 96;rr(x,y,w,h,f,s);tx(x+10,y+h-25,t,9,INK,'Helvetica-Bold');para(x+10,y+h-43,b,7.5,w-20,9,MUTED,4)
for left,right in zip(flow,flow[1:]):arrow(left[0]+left[2]+3,553,right[0]-4,553)
rr(695,488,116,54,BLUE,BLUEB,6);tx(706,521,'Exact classical',7.8,INK,'Helvetica-Bold');para(706,507,'Correctness and cost control.',7,96,8,MUTED,2)
rr(827,488,116,54,TEAL,TEALB,6);tx(838,521,'QAOA',7.8,INK,'Helvetica-Bold');para(838,507,'p=2 statevector search.',7,96,8,MUTED,2);tx(814,507,'OR',6.8,MUTED,'Helvetica-Bold')
c.setStrokeColor(NAVY);c.setLineWidth(1.8);c.line(1042,501,1042,450);c.line(1042,450,104,450);arrow(104,450,104,501,NAVY,1.8);ctr(350,435,520,'Observation changes the next objective; repeat until solved or budget exhausted',8,NAVY)

# implementation boxes
tx(42,400,'WHAT IS IMPLEMENTED',10,NAVY,'Helvetica-Bold')
items=[(42,285,286,96,'Factorized world model','factorized.py + rule_environment.py','Player, Goal, Movement, Topology, Interaction and State. Two options each produce 2^6 = 64 complete world models.',BLUE,BLUEB),(344,285,286,96,'Compact optimization model','factorized_objective.py + qubo.py','Information gain comes from six marginals. QUBO uses six hypothesis bits and seven one-hot action variables.',GOLD,GOLDB),(646,285,286,96,'Matched experiment runner','factorized_agent.py + benchmark.py','Policy proxy, exact classical search and QAOA receive identical states, utilities and action choices.',TEAL,TEALB),(948,285,306,96,'Provider and environment slots','providers/ + environment_adapter.py','OpenAI Responses API adapter reads environment variables. External ARC or domain objects can be wrapped later.',LILAC,LILACB)]
for x,y,w,h,t,f,b,fill,stroke in items:
 rr(x,y,w,h,fill,stroke);tx(x+12,y+h-26,t,9,INK,'Helvetica-Bold');tx(x+12,y+h-43,f,7.5,stroke,'Helvetica-Bold');para(x+12,y+h-59,b,7.4,w-24,8.5,MUTED,4)

# results and scale
rr(42,105,588,156,GRAY,LINE);tx(57,237,'PRELIMINARY ENGINEERING RUNS',9,NAVY,'Helvetica-Bold')
rows=[('Policy proxy','64','100%','7.000','~0.007 s'),('Exact classical','64','100%','7.000','~0.007 s'),('QAOA smoke','8','100%','6.875','~3.26 s')]
headers=['Condition','Worlds','Solve rate','Mean actions','Mean wall time'];xs=[57,220,300,390,500]
for x,h in zip(xs,headers):tx(x,216,h,7.5,INK,'Helvetica-Bold')
c.setStrokeColor(LINE);c.line(57,209,615,209)
for r,row in enumerate(rows):
 y=191-r*22
 for x,value in zip(xs,row):tx(x,y,value,7.8,INK)
para(57,121,'Smoke tests validate the implementation only. The QAOA configuration was developed on small samples; no advantage claim is justified.',7.4,548,9,RED,3)

rr(646,105,608,156,GRAY,LINE);tx(661,237,'SCALING ESTIMATE FOR CURRENT ONE-HOT ACTION FORMULATION',9,NAVY,'Helvetica-Bold')
headers=['Hypotheses','H bits','Total qubits','Statevector','Raw memory'];xs=[661,770,842,947,1080]
for x,h in zip(xs,headers):tx(x,216,h,7.5,INK,'Helvetica-Bold')
c.line(661,209,1238,209)
scale=[('64','6','13','8,192','128 KiB'),('256','8','15','32,768','512 KiB'),('262,144','18','25','33.6M','512 MiB'),('16,777,216','24','31','2.15B','32 GiB')]
for r,row in enumerate(scale):
 y=191-r*21
 for x,value in zip(xs,row):tx(x,y,value,7.7,INK)
para(661,112,'Memory excludes simulator overhead, ancillas and transpilation. Real feasibility may fail before the raw qubit count.',7.4,570,9,RED,2)
tx(42,78,'CURRENT BOUNDARY',8,NAVY,'Helvetica-Bold');tx(42,60,'The QUBO is compact, but statevector simulation still expands its complete cost diagonal. This is a correctness prototype, not evidence of quantum speedup.',8.2,RED,'Helvetica-Bold')
tx(42,31,'Raw outputs: output/experiments/  |  Method notes: EXPERIMENT_NOTES.md  |  API key is never stored in source or traces.',8.2,MUTED)
c.showPage()

# PAGE 2
header('Complete file map and responsibilities','Page 2 of 2')
tx(42,620,'RESEARCH PATH',10,NAVY,'Helvetica-Bold')
cols=[
(42,344,286,252,'64-HYPOTHESIS MODEL',BLUE,BLUEB,[('src/qahs/factorized.py','Six rule dimensions, actions, hypotheses, states and traces.'),('src/qahs/rule_environment.py','Controlled hidden-world environment for matched experiments.'),('src/qahs/environment_adapter.py','Contract and wrapper for future ARC or domain environments.'),('src/qahs/factorized_objective.py','Marginal information gain, action scores and belief updates.'),('src/qahs/qubo.py','QUBO model, Ising conversion, constraints and resource counts.'),('src/qahs/factorized_agent.py','Observe-search-act-update loop for the 64-world study.')]),
(344,344,286,252,'EXPERIMENTS + SCALE',TEAL,TEALB,[('src/qahs/benchmark.py','Repeated matched runs and aggregate metrics.'),('src/qahs/benchmark_cli.py','Command-line benchmark entry point.'),('src/qahs/scaling.py','Qubit, amplitude and statevector-memory estimates.'),('output/experiments/*.json','Preserved row-level results and summaries.'),('output/experiments/README.md','Plain-language index of result files.'),('EXPERIMENT_NOTES.md','Interpretation, tuning history and remaining requirements.')]),
(646,344,286,252,'SEARCH + PROVIDERS',GOLD,GOLDB,[('src/qahs/search/base.py','Shared search backend contract.'),('src/qahs/search/greedy.py','Immediate-utility policy proxy.'),('src/qahs/search/classical.py','Exact minimum over valid encoded states.'),('src/qahs/search/qaoa.py','Cost phase, X mixer, COBYLA and sampled decoding.'),('src/qahs/providers/openai_provider.py','Optional Responses API caller using environment credentials.'),('.env.example','Variable names only; never contains a real key.')]),
(948,344,306,252,'INTERFACES + VERIFICATION',LILAC,LILACB,[('dashboard/server.py','Dependency-free local run dashboard and JSON endpoint.'),('dashboard/app.py','Optional Streamlit run/replay interface.'),('tests/test_qahs.py','Nine checks for encoding, objectives, QUBO/Ising and loops.'),('pyproject.toml','Package, scripts and optional dependency groups.'),('README.md','Setup, commands, boundaries and project map.'),('.gitignore','Excludes credentials, caches and generated traces.')])]
for x,y,w,height,heading,fill,stroke,files in cols:
 rr(x,y,w,height,fill,stroke);tx(x+12,y+225,heading,9,stroke,'Helvetica-Bold');yy=y+201
 for name,purpose in files:yy=fileline(x+12,yy,name,purpose,w-24)

tx(42,312,'FOUNDATIONAL THREE-QUBIT DEMO',10,NAVY,'Helvetica-Bold')
cols2=[(42,114,388,176,'DOMAIN + LOOP',BLUE,BLUEB,[('src/qahs/models.py','Original actions, states, hypotheses, scores and trace records.'),('src/qahs/environment.py','Red-tile switch demonstration.'),('src/qahs/interpreter.py','Deterministic and structured-LLM interpreter interfaces.'),('src/qahs/compiler.py','Binary encode/decode for hypothesis-action pairs.'),('src/qahs/agent.py','Original two-turn harness.')]),(454,114,388,176,'PREDICTION + STORAGE',TEAL,TEALB,[('src/qahs/predictor.py','Predicted observations and progress values.'),('src/qahs/objective.py','Entropy, information gain and posterior updates.'),('src/qahs/run_store.py','Replayable JSON trace storage.'),('src/qahs/cli.py','Three-qubit command-line runner.'),('src/qahs/__init__.py','Public package exports.')]),(866,114,388,176,'ARTIFACTS + SOURCES',GRAY,LINE,[('output/pdf/','Current codebase overview.'),('tmp/pdfs/','Reproducible PDF build source.'),('Instructor QAOA notebook','Source pattern for phase, mixer, COBYLA and sampling.'),('Not reused','Malware graph, similarity weights and Max-Cut objective.'),('Dashboard URL','http://127.0.0.1:8000')])]
for x,y,w,height,heading,fill,stroke,files in cols2:
 rr(x,y,w,height,fill,stroke);tx(x+12,y+150,heading,9,stroke if stroke!=LINE else NAVY,'Helvetica-Bold');yy=y+127
 for name,purpose in files:yy=fileline(x+12,yy,name,purpose,w-24)
c.setStrokeColor(LINE);c.line(42,93,W-42,93)
tx(42,70,'RUN CONTROLS',8,NAVY,'Helvetica-Bold');tx(42,52,'qahs-demo: 3-qubit loop  |  qahs-benchmark: 64-hypothesis matched study  |  dashboard/server.py: local browser interface',8.3,INK,'Helvetica-Bold')
tx(42,27,'Next gate: real LLM + genuine ARC tasks + pre-registered held-out evaluation. Current runs cannot establish stronger AI or quantum advantage.',8.2,RED,'Helvetica-Bold')
c.showPage();c.save();print(OUT)
