// Eight independent, bounded numerical explainers. No external assets or auto-play.
const VIZ_META={
 projection:['一个矩阵格子怎样算出来','TH03'],attention:['注意力：打分、遮罩与加权','TH04'],
 heads:['拆头与合头：追踪数字的位置','TH02'],block:['一行向量怎样穿过 Transformer','TH10'],
 rope:['RoPE：位置改变角度','TH18'],cache:['KV Cache：哪些计算可以复用','TH17'],
 probability:['交叉熵与 KL：猜准了还扣分吗','M02'],lora:['LoRA：小旁路怎样修改输出','SFT03']
};
const QUESTION_VIZ={TH02:'heads',TH03:'projection',TH04:'attention',TH05:'attention',TH06:'heads',TH10:'block',TH13:'block',TH15:'probability',TH17:'cache',TH18:'rope',TH19:'cache',M02:'probability',T01:'block',T02:'attention',SFT03:'lora',KD03:'probability'};
function visualPanel(id){const type=QUESTION_VIZ[id];return type?'<details class="viz-panel" data-viz="'+type+'"><summary>动手看：'+VIZ_META[type][0]+'</summary><div class="viz-host"></div></details>':'';}
function vizNum(n,d=3){return Math.abs(n)<1e-10?'0':Number(n.toFixed(d)).toString();}
function vizSoftmax(row){const m=Math.max(...row),values=row.map(x=>Math.exp(x-m)),sum=values.reduce((a,b)=>a+b,0);return values.map(x=>x/sum);}
function vizDot(a,b){return a.reduce((sum,x,i)=>sum+x*b[i],0);}
function vizMatmul(a,b){return a.map(row=>b[0].map((_,j)=>row.reduce((sum,x,k)=>sum+x*b[k][j],0)));}
function vizMatrix(name,matrix,options={}){
 const {select=false,row=0,col=0,headers=null,masked=null,heat=false}=options;
 return '<div class="viz-table-wrap" tabindex="0" role="group" aria-label="'+name+'；可横向滚动"><table class="viz-matrix"><caption>'+name+'</caption><thead><tr><th scope="col">行 / 列</th>'+matrix[0].map((_,i)=>'<th scope="col">'+(headers?headers[i]:i)+'</th>').join('')+'</tr></thead><tbody>'+matrix.map((r,i)=>'<tr><th scope="row">'+i+'</th>'+r.map((v,j)=>{
  const hidden=masked&&!masked[i][j],val=hidden?'遮住':vizNum(v),style=heat&&!hidden?' style="background:rgba(31,100,75,'+(0.04+0.5*v)+')"':'';
  return '<td'+(hidden?' class="viz-hidden"':'')+style+'>'+(select?'<button type="button" data-cell="'+i+','+j+'" aria-label="第 '+i+' 行第 '+j+' 列，'+val+'" aria-pressed="'+(row===i&&col===j)+'">'+val+'</button>':val)+'</td>';
 }).join('')+'</tr>').join('')+'</tbody></table></div>';
}
function vizRange(key,label,min,max,value,step=1){return '<label>'+label+' <output data-output="'+key+'">'+value+'</output><input aria-label="'+label+'" data-key="'+key+'" type="range" min="'+min+'" max="'+max+'" step="'+step+'" value="'+value+'"></label>';}
function vizCheck(key,label,value){return '<label><input type="checkbox" data-key="'+key+'"'+(value?' checked':'')+'>'+label+'</label>';}
function vizSelect(key,label,options,value){return '<label>'+label+' <select data-key="'+key+'" aria-label="'+label+'">'+options.map(([v,t])=>'<option value="'+v+'"'+(String(v)===String(value)?' selected':'')+'>'+t+'</option>').join('')+'</select></label>';}
function vizBar(label,value,max,secondary=false,suffix=''){return '<div class="viz-bar-row"><span>'+label+'</span><div class="viz-track"><div class="viz-fill'+(secondary?' secondary':'')+'" style="width:'+Math.min(100,100*value/max)+'%"></div></div><span>'+vizNum(value)+suffix+'</span></div>';}
function vizDetail(text){return '<div class="viz-detail" role="status" aria-live="polite">'+text+'</div>';}
function mountVisual(host,type){
 if(host.dataset.mounted)return;
 host.dataset.mounted=type;
 const state={row:0,col:0,head:0,causal:true,scaled:true,stage:'weights',heads:2,step:0,m:1,n:3,freq:1,decode:1,p:80,q:60,rank:1,alpha:2};
 let controls='',description='';
 if(type==='projection')description='示意输入和权重，数值真实相乘。点击 Q 的一个格子，追踪输入哪一行与权重哪一列参与计算。';
 if(type==='attention'){
  description='4 个位置、每头 2 维的教学数据。先看打分，再开关因果遮罩；点击任一格，下面显示计算路径。';
  controls=vizSelect('head','示意头',[[0,'头 0'],[1,'头 1']],0)+vizCheck('causal','因果遮罩',true)+vizCheck('scaled','除以 √Dh',true)+vizSelect('stage','矩阵显示',[['scores','缩放分数'],['weights','softmax 权重']],'weights');
 }
 if(type==='heads'){description='每个词原有 8 个数。只拆分通道、调整轴顺序，数字与词的对应不能乱。固定 3 个词，B=1。';controls=vizSelect('heads','头数',[[1,'1 头'],[2,'2 头'],[4,'4 头']],2);}
 if(type==='block'){description='跟随一段长度为 3 的输入，查看 Pre-LN 小语言模型的形状和对应代码。这是结构图，不是模型激活值。';controls=vizRange('step','当前步骤',0,6,0);}
 if(type==='rope'){description='只展示一对通道：初始 Q=K=[1,0]。转角=位置×频率；点积会感受到相对位置。其余通道对使用其他频率。';controls=vizRange('m','Q 位置',0,6,1)+vizRange('n','K 位置',0,6,3)+vizSelect('freq','本对频率',[[1,'1 弧度/位置'],[0.1,'0.1 弧度/位置']],1);}
 if(type==='cache'){description='提示有 3 个 token；拖动滑块追加生成步骤。比较已完成前向的累计打分矩阵格数，含因果遮住的格子。它不是实际耗时倍数。';controls=vizRange('decode','追加解码步数',0,5,1);}
 if(type==='probability'){description='两类事件：红球与蓝球。P 是真实比例，Q 是预测比例；下方扣分用自然对数，单位 nat。改变 Q，观察它贴近 P 时会发生什么。';controls=vizRange('p','真实红球比例 %',1,99,80)+vizRange('q','预测红球比例 %',1,99,60);}
 if(type==='lora'){description='原矩阵 W₀ 冻结，旁路为 (α/r)·B·A·x。示意 x 与矩阵真实计算；改变秩或 α，看旁路参数量与输出怎样变化。';controls=vizSelect('rank','秩 r',[[1,'1'],[2,'2'],[3,'3']],1)+vizRange('alpha','α',0,4,2,0.5);}
 host.innerHTML='<h3>'+VIZ_META[type][0]+'</h3><p class="viz-note">'+description+'</p><div class="viz-controls">'+controls+'</div><div class="viz-result"></div><p class="viz-note"><a href="#q-'+VIZ_META[type][1]+'">对应讲解与手写代码 →</a></p>';
 const result=host.querySelector('.viz-result');
 const draw=()=>{
  const active=document.activeElement?.dataset.cell;
  if(type==='projection'){
   const x=[[1,2,-1],[0,1,2],[2,-1,0]],w=[[1,0],[2,-1],[-1,2]],y=vizMatmul(x,w),i=state.row,j=state.col;
   result.innerHTML='<div class="viz-grid">'+vizMatrix('输入 X [3,3]',x)+vizMatrix('计算方向 Wᵀ [3,2]',w)+vizMatrix('输出 Q = XWᵀ [3,2]，点击格子',y,{select:true,row:i,col:j})+'</div>'+vizDetail('Q['+i+','+j+'] = '+x[i].map((a,k)=>'('+a+' × '+w[k][j]+')').join(' + ')+' = <strong>'+y[i][j]+'</strong>。这里省略 bias；nn.Linear 实际存储 W [2,3]。');
  }
  if(type==='attention'){
   const q=state.head===0?[[1,0],[0,1],[1,1],[-1,1]]:[[0,1],[1,-1],[1,0],[1,1]];
   const k=state.head===0?[[1,0],[0,1],[1,1],[1,-1]]:[[1,1],[0,1],[1,-1],[-1,0]];
   const v=[[2,0],[0,4],[1,1],[-1,2]],scale=state.scaled?Math.sqrt(2):1;
   const scores=q.map(a=>k.map(b=>vizDot(a,b)/scale)),allowed=scores.map((r,i)=>r.map((_,j)=>!state.causal||j<=i));
   const weights=scores.map((r,i)=>vizSoftmax(r.map((s,j)=>allowed[i][j]?s:-Infinity)));
   const y=vizMatmul(weights,v),i=state.row,j=state.col,shown=state.stage==='scores'?scores:weights;
   const math=allowed[i][j]?'Q·K = '+q[i].map((a,z)=>'('+a+'×'+k[j][z]+')').join(' + ')+'；分数='+vizNum(scores[i][j])+ '；softmax 权重='+vizNum(weights[i][j],5)+'。':'该列在当前位置的未来，被遮住后权重为 0。';
   result.innerHTML='<div class="viz-grid">'+vizMatrix('Q [4,2]',q)+vizMatrix('K [4,2]',k)+'</div>'+vizMatrix(state.stage==='scores'?'缩放后分数：行=query，列=key':'注意力权重：行=query，列=key',shown,{select:true,row:i,col:j,masked:allowed,heat:state.stage==='weights'})+vizDetail('位置 '+i+' 看位置 '+j+'：'+math+'本行权重和='+vizNum(weights[i].reduce((a,b)=>a+b),6)+'。')+'<div class="viz-grid">'+vizMatrix('V：准备取走的内容',v)+vizMatrix('输出 = 权重 × V',y)+'</div><p class="viz-note">当前行输出：['+v[0].map((_,c)=>weights[i].map((a,t)=>vizNum(a)+'×'+v[t][c]).join(' + ')+' = '+vizNum(y[i][c])).join('；')+']</p>';
  }
  if(type==='heads'){
   const x=Array.from({length:3},(_,t)=>Array.from({length:8},(_,d)=>8*t+d)),h=state.heads,dh=8/h;
   result.innerHTML=vizMatrix('原输入 [1,3,8]：行=词位置',x)+'<div class="viz-flow"><div class="viz-node"><strong>[1,3,8]</strong><span>原向量</span></div><div class="viz-node"><strong>[1,3,'+h+','+dh+']</strong><span>reshape 拆通道</span></div><div class="viz-node active"><strong>[1,'+h+',3,'+dh+']</strong><span>transpose 换轴</span></div></div><div class="viz-grid">'+Array.from({length:h},(_,head)=>vizMatrix('头 '+head+'：仍有 3 个词',x.map(r=>r.slice(head*dh,(head+1)*dh)))).join('')+'</div>'+vizDetail('合头先交换回 [1,3,'+h+','+dh+']，再合并最后两维。第 1 个词在每个头的片段合起来，仍是 '+x[1].join('、')+'。');
  }
  if(type==='block'){
   const steps=[['查表与位置','[1,3] → [1,3,8]','x = token_emb(ids) + pos_emb(pos)','编号变向量，加入位置信息。'],['第一次归一化','[1,3,8] → [1,3,8]','u = norm1(x)','只调整每个词的通道数值，不改形状。'],['多头注意力','[1,2,3,4] → [1,3,8]','a = out_proj(merge_heads(weights @ v))','两个头各看前文，合头后再混合。'],['第一次残差','[1,3,8] + [1,3,8]','x = x + a','加回进入子层之前的 x。'],['第二次归一化与 FFN','8 → 32 → 8','f = down(gelu(up(norm2(x))))','逐位置加工特征。'],['第二次残差','[1,3,8] + [1,3,8]','x = x + f','这一层完成；下一层有自己的参数。'],['最后预测词表','[1,3,8] → [1,3,10]','logits = lm_head(final_norm(x))','这里示意词表 10；选最后位置用于续写。']];
   result.innerHTML='<div class="viz-flow">'+steps.map((s,i)=>'<div class="viz-node'+(i===state.step?' active':'')+'"><strong>'+(i+1)+'. '+s[0]+'</strong><span>'+s[1]+'</span></div>').join('')+'</div>'+vizDetail('<strong>'+steps[state.step][0]+'</strong><br><code>'+escapeHtml(steps[state.step][2])+'</code><br>'+steps[state.step][3]);
  }
  if(type==='rope'){
   const a=state.m*state.freq,b=state.n*state.freq,q=[Math.cos(a),Math.sin(a)],k=[Math.cos(b),Math.sin(b)],cx=160,cy=145,r=92;
   const vector=(v,label,color,y)=>'<line x1="'+cx+'" y1="'+cy+'" x2="'+(cx+r*v[0])+'" y2="'+(cy-r*v[1])+'" stroke="'+color+'" stroke-width="3"/><circle cx="'+(cx+r*v[0])+'" cy="'+(cy-r*v[1])+'" r="5" fill="'+color+'"/><text x="18" y="'+y+'">'+label+'=['+v.map(n=>vizNum(n)).join(', ')+']</text>';
   result.innerHTML='<svg class="viz-svg" viewBox="0 0 320 330" role="img" aria-label="Q 与 K 单位向量随位置旋转，半径保持 1"><circle cx="160" cy="145" r="92" fill="none" stroke="#b8c7bc"/><line x1="40" y1="145" x2="280" y2="145" stroke="#b8c7bc"/><line x1="160" y1="35" x2="160" y2="255" stroke="#b8c7bc"/><text x="270" y="164">x</text><text x="170" y="39">y</text>'+vector(k,'K（橙）','#ad5d29',305)+vector(q,'Q（绿）','#1f644b',282)+'</svg>'+vizDetail('Q·K = cos(('+state.m+' − '+state.n+') × '+state.freq+') = <strong>'+vizNum(vizDot(q,k),6)+'</strong>。两者长度都为 1；同样平移位置保持相对角度。');
  }
  if(type==='cache'){
   const n=3+state.decode;let plain=9,cached=9;
   for(let t=4;t<=n;t++){plain+=t*t;cached+=t;}
   result.innerHTML='<div class="viz-flow">'+Array.from({length:n},(_,i)=>'<span class="viz-token'+(i>=3?' new':'')+'">位置 '+i+'</span>').join('')+'</div><div class="viz-grid"><div><h4>每次重新算整段</h4>'+vizMatrix('本次 Q×Kᵀ 的形状',[[n,n]],{headers:['query 数','key 数']})+'</div><div><h4>复用历史 K/V</h4>'+vizMatrix('本次 Q×Kᵀ 的形状',[[state.decode?1:3,n]],{headers:['query 数','key 数']})+'</div></div><div class="viz-bars">'+vizBar('重复算格数',plain,plain)+vizBar('缓存后格数',cached,plain,true)+'</div>'+vizDetail('累计 '+plain+' vs '+cached+' 个打分格。2 层、1 样本、2 个 KV 头、每头 4 维：缓存 '+(32*n)+' 个元素；float32 为 '+(128*n)+' 字节。新 Q 仍读取全部 '+n+' 个 K。')+'<p class="viz-note">仅计算密集打分矩阵格数，未包含 FFN、投影、带宽、缓存管理等开销。CPU 小模型会另检验完整与缓存 logits 相等。</p>';
  }
  if(type==='probability'){
   const p=state.p/100,q=state.q/100,h=-p*Math.log(p)-(1-p)*Math.log(1-p),ce=-p*Math.log(q)-(1-p)*Math.log(1-q),kl=ce-h;
   result.innerHTML='<div class="viz-bars">'+vizBar('真实红球 P',p,1,false)+vizBar('预测红球 Q',q,1,true)+'</div>'+vizMatrix('扣分分解（自然对数，nat）',[[h,ce,kl]],{headers:['熵 H(P)','交叉熵 H(P,Q)','KL(P||Q)']})+vizDetail('交叉熵 = −'+vizNum(p)+'·ln('+vizNum(q)+') − '+vizNum(1-p)+'·ln('+vizNum(1-q)+') = '+vizNum(ce,6)+'。<br>额外扣分 KL = '+vizNum(ce,6)+' − '+vizNum(h,6)+' = <strong>'+vizNum(kl,6)+'</strong>。')+'<p>当 Q=P 时，KL 为 0，但两种颜色仍有不确定性，所以交叉熵仍大于 0。这里展示的是完整分布的平均扣分；单次抽到红球只扣 −ln(Q红)。</p>';
  }
  if(type==='lora'){
   const r=state.rank,x=[1,2,-1,.5],w=[[1,0,1,0],[0,1,0,1],[1,-1,0,1]],A=Array.from({length:r},(_,i)=>[1,.5*(i+1),-1,.25*i]),B=Array.from({length:3},(_,o)=>Array.from({length:r},(_,i)=>(o+1)*(i%2?-.2:.3)));
   const ax=A.map(row=>vizDot(row,x)),base=w.map(row=>vizDot(row,x)),delta=B.map(row=>state.alpha/r*vizDot(row,ax)),out=base.map((v,i)=>v+delta[i]);
   result.innerHTML='<div class="viz-flow"><div class="viz-node"><strong>x [4]</strong><span>'+x.join(', ')+'</span></div><div class="viz-node"><strong>A·x ['+r+']</strong><span>'+ax.map(v=>vizNum(v)).join(', ')+'</span></div><div class="viz-node active"><strong>(α/r) B·A·x [3]</strong><span>'+delta.map(v=>vizNum(v)).join(', ')+'</span></div></div><div class="viz-grid">'+vizMatrix('A ['+r+',4]',A)+vizMatrix('B [3,'+r+']',B)+'</div>'+vizMatrix('每个输出通道：冻结主路 + 旁路',base.map((v,i)=>[v,delta[i],out[i]]),{headers:['W₀x','旁路修正','最终输出']})+vizDetail('主矩阵 3×4=12 个参数；旁路 r×(4+3)=<strong>'+(7*r)+'</strong> 个可训参数。这个小例子 r≥2 时旁路参数已超过主矩阵，说明低秩必须相对原维度足够小才省参数。α=0 时输出回到主路。');
  }
  if(active){const button=result.querySelector('[data-cell="'+active+'"]');button?.focus({preventScroll:true});}
  for(const out of host.querySelectorAll('[data-output]'))out.value=state[out.dataset.output];
 };
 host.addEventListener('input',e=>{const el=e.target,key=el.dataset.key;if(!key)return;state[key]=el.type==='checkbox'?el.checked:el.tagName==='SELECT'&&key==='stage'?el.value:Number(el.value);draw();});
 result.addEventListener('click',e=>{const cell=e.target.closest('[data-cell]');if(!cell)return;[state.row,state.col]=cell.dataset.cell.split(',').map(Number);draw();});
 draw();
}
content.addEventListener('toggle',e=>{if(e.target.open&&content.contains(e.target)&&e.target.matches('.viz-panel'))mountVisual(e.target.querySelector('.viz-host'),e.target.dataset.viz);},true);
function visualizationPage(type='attention'){
 if(!VIZ_META[type])type='attention';
 content.innerHTML='<article class="article"><h1>算法可视化：先预测，再动手验证</h1><p>每次只看一个小实验。改变参数、点击格子，然后回到对应代码，把看到的规律写出来。</p><div class="viz-picker">'+Object.entries(VIZ_META).map(([id,[title]])=>'<button class="btn" data-route="viz-'+id+'" aria-current="'+(type===id)+'">'+title.split('：')[0]+'</button>').join('')+'</div><div class="viz-host" id="visualStage"></div></article>';
 for(const b of content.querySelectorAll('[data-route]'))b.onclick=()=>go(b.dataset.route);
 mountVisual(document.getElementById('visualStage'),type);
}
function transformerOverview(){return '<h1>Transformer 手写：22 步从形状到模型</h1><p>先用 2 句话、3 个词、8 个通道讲清每一步，再写出能前向、训练和生成的小模型。每题包括代码、练习、答案与追问。</p><p><a href="#q-TH01">从第 1 步开始 →</a> · <a href="#viz-attention">先动手看注意力</a> · <a href="#q-TH21">30 分钟手写练习</a></p><div class="viz-flow"><div class="viz-node"><strong>1—5 形状与注意力</strong><span>看懂一个矩阵格子</span></div><div class="viz-node"><strong>6—14 拼成架构</strong><span>多头、Block、两种完整模型</span></div><div class="viz-node"><strong>15—20 学习与验证</strong><span>loss、训练、缓存与现代部件</span></div><div class="viz-node"><strong>21—22 面试与迁移</strong><span>合上答案再写一遍</span></div></div><p><button class="btn" data-code="transformer_handwrite.py">下载完整实现</button> <button class="btn" data-code="test_transformer_handwrite.py">下载测试</button> <button class="btn" data-code="Transformer手写运行与练习.md">下载运行说明</button></p><p class="small">本页无需安装环境；运行下载的 Python 代码需要 PyTorch。小模型学习的是玩具循环序列，现代部件另有独立测试。</p>';}
let codeAssets=null;
function downloadCode(name){if(!codeAssets)codeAssets=JSON.parse(document.getElementById('code-assets').textContent);if(!Object.hasOwn(codeAssets,name))return;const blob=new Blob([codeAssets[name]],{type:'text/plain;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1500);}
content.addEventListener('click',e=>{const b=e.target.closest('[data-code]');if(b){downloadCode(b.dataset.code);return;}const a=e.target.closest('a');if(a){const name=a.getAttribute('href')?.split('/').pop();if(['transformer_handwrite.py','test_transformer_handwrite.py'].includes(name)){e.preventDefault();downloadCode(name);}}});
