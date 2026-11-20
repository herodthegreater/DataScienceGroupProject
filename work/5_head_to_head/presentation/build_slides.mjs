import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const sourceDir=path.dirname(fileURLToPath(import.meta.url));
const base=path.resolve(process.env.SLIDES_BUILD_DIR || path.join(sourceDir,'build'));
await fs.mkdir(base,{recursive:true});
const data=JSON.parse(await fs.readFile(path.join(sourceDir,'data.json'),'utf8'));
const p=Presentation.create({slideSize:{width:1280,height:720}});
p.theme.defaultFont='Avenir Next';
const C={black:'#121212',sage:'#EAF2E8',forest:'#182D25',white:'#F4F4F0',gray:'#A9ABA5',green:'#1ED760',mint:'#8BF1B1',deep:'#077746'};
const genreOrder=['electronic','rock','pop','latin','hiphop/rnb'];
const genreNames=['Electronic','Rock','Pop','Latin','Hip-hop / R&B'];
const names={baseline:'Baseline',logistic_regression:'Logistic',random_forest:'Forest'};
const om=Object.fromEntries(data.overall_metrics.map(r=>[r.model,r]));
const source=`Sources: https://github.com/herodthegreater/DataScienceGroupProject/tree/${data.audit.repository_commit}\n`+
  `Official predictions: predictions/logistic_regression.csv and predictions/random_forest.csv\n`+
  `Analysis: work/5_head_to_head/evaluate.py and results/tables/*.csv\n`+
  `Both files have 9,009 unique track IDs, the same truth and metadata, valid named class probabilities, and predicted labels consistent with probability maxima.\n`+
  `Model order: ${data.audit.classes_probability_order.join(', ')}.\n`+
  `Dataset SHA-256: ${data.audit.files_sha256['clean_songs.csv']}\n`+
  `Reference design and original Spotify wordmark: user-supplied Audio Atlas presentation.\n`;

function txt(slide,text,x,y,w,h,pt=18,color=C.white,bold=false) {
 const shape=slide.shapes.add({geometry:'textbox',name:text.slice(0,60),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 shape.text=text;
 shape.text.style={fontSize:`${pt}pt`,color,bold,verticalAlignment:'top',wrap:'word',insets:0};
 shape.text.insets=0;
 return shape;
}
async function header(title,subtitle,bg,section=''){
 const slide=p.slides.add();slide.background.fill=bg;
 const dark=bg!==C.sage;
 const logo=await fs.readFile(path.join(sourceDir,'assets',dark?'spotify-white.png':'spotify-black.png'));
 slide.images.add({blob:logo.buffer.slice(logo.byteOffset,logo.byteOffset+logo.byteLength),contentType:'image/png',alt:'Spotify',fit:'contain',position:{left:1093,top:45,width:115,height:31.5}});
 if(section)txt(slide,section,72,45,830,28,12.75,dark?C.gray:'#445448',true);
 txt(slide,title,72,91,1136,78,42,dark?C.white:C.black,true);
 txt(slide,subtitle,72,170,1136,34,16,dark?C.gray:'#445448');
 return slide;
}
function table(slide,values,x,y,w,h,widths,dark=true,font=17){
 const t=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
 t.styleOptions={headerRow:false,bandedRows:false,firstColumn:false};
 t.borders.assign({fill:dark?'#34463B':'#C6D6CB',width:0.6});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=h/values.length;
  for(let c=0;c<values[r].length;c++){
   const cell=t.getCell(r,c);
   cell.fill=r===0?(dark?'#243A2E':'#DDE9DC'):(dark?C.black:C.sage);
   cell.text.style={fontSize:`${r===0?14:font}pt`,color:r===0?(dark?C.gray:'#445448'):(c===values[r].length-1?(dark?C.mint:C.deep):(dark?C.white:C.black)),bold:r===0||c===values[r].length-1,alignment:c===0?'left':'right',verticalAlignment:'middle'};
  }
 }
 return t;
}

const s1=await header('What 300 trees buy us','The same 9,009 test songs, using the saved model outputs',C.black,'ISC 4241  /  HEAD TO HEAD');
txt(s1,'+7.1 pts',66,267,460,128,78,C.green,true);
txt(s1,'Accuracy gain',72,410,420,45,22,C.white,true);
txt(s1,'637 more songs correct',72,466,430,40,19,C.white);
txt(s1,'Forest only: 1,221 correct\nLogistic only: 584 correct',72,520,435,70,17.5,C.gray);
const metrics=[['Accuracy','accuracy'],['Macro precision','macro_precision'],['Macro recall','macro_recall'],['Macro F1','macro_f1'],['ROC-AUC','roc_auc_ovr_macro']];
table(s1,[['Metric','Baseline','Logistic','Forest'],...metrics.map(([label,key])=>[label,...['baseline','logistic_regression','random_forest'].map(m=>key==='accuracy'?`${(om[m][key]*100).toFixed(2)}%`:om[m][key].toFixed(3))])],545,252,663,312,[240,126,145,152]);
txt(s1,'The baseline always predicts Electronic, the largest training genre.',545,586,663,50,14,C.gray);
txt(s1,'ROC-AUC averages five one-vs-rest scores. Macro metrics give each genre equal weight.',72,650,1136,36,12.75,C.gray);
const spoken1=`My question is what Random Forest's extra complexity actually buys us. I compared both saved prediction files on exactly the same 9,009 test songs, matching them by track ID and checking the labels and probabilities. The baseline always predicts Electronic, the largest training genre. It gets 30.3 percent right. Logistic Regression reaches 56.95 percent, and Random Forest reaches 64.02 percent. That is a gain of 7.1 percentage points, or 637 additional correct songs. The paired comparison shows why this is a net gain. The forest gets 1,221 songs right that logistic regression misses, but logistic regression gets 584 right that the forest misses. Macro F1 also improves from 0.489 to 0.556, so the gain survives when every genre gets equal weight.`;
s1.speakerNotes.text=`ABOUT 60 SECONDS\n${spoken1}\n\nNOT SPOKEN\n${source}The majority class is selected from training data only. ROC-AUC is macro one-vs-rest. No hyperparameters were selected using test results. Paired accuracy gain: 637 / 9009 = 0.070707. Artist-credit cluster bootstrap 95% interval: +5.24 to +8.81 percentage points, conditional on this split.\n`;

const s2=await header('Every genre gains F1','A stronger overall model still misses most hip-hop / R&B songs',C.sage);
s2.charts.add('bar',{
 position:{left:64,top:230,width:765,height:346},categories:[...genreNames].reverse(),
 series:['logistic_regression','random_forest'].map((m,i)=>({name:i?'Random Forest':'Logistic Regression',values:[...genreOrder].reverse().map(g=>Number(data.per_genre_metrics.find(r=>r.model===m&&r.genre===g).f1.toPrecision(12))),fill:i?C.deep:C.gray,valuesFormatCode:'0.000'})),
 barOptions:{direction:'bar',grouping:'clustered',gapWidth:65,overlap:0},hasLegend:true,
 legend:{position:'bottom',overlay:false,textStyle:{typeface:'Avenir Next',fontSize:'13pt',fill:C.black}},
 xAxis:{textStyle:{typeface:'Avenir Next',fontSize:'14pt',fill:C.black},line:{fill:'none',width:0},majorGridlines:null},
 yAxis:{min:0,max:.9,majorUnit:.3,numberFormatCode:'0.0',textStyle:{typeface:'Avenir Next',fontSize:'13pt',fill:'#445448'},line:{fill:'none',width:0},majorGridlines:{fill:'#CCD9CF',width:.7}},
 dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{typeface:'Avenir Next',fontSize:'13pt',fill:C.black}},
 chartFill:C.sage,plotAreaFill:C.sage,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0}
});
txt(s2,'12.9%',884,272,324,110,65,C.deep,true);
txt(s2,'Hip-hop / R&B recall',888,393,320,66,20,C.black,true);
txt(s2,'49 of 381 songs found\nLogistic Regression finds 38',888,472,320,100,17.5,'#445448');
txt(s2,'Higher accuracy comes with 300 trees and 1.78 million nodes',72,600,1136,44,18,C.black,true);
txt(s2,'F1 balances precision and recall. Artist groups use full credit strings, so collaborators can cross the split.',72,654,1136,32,12.75,'#445448');
const spoken2=`Random Forest improves F1 in all five genres, with Latin gaining the most. Hip-hop and R&B stays the blind spot. The forest finds 49 of 381 songs, compared with 38 for logistic regression, and still misses 332. Our common feature test shows stronger forest reliance on acousticness and instrumentalness. Both models rely strongly on danceability. The gain comes with 300 trees and about 1.78 million nodes, compared with 70 logistic weights plus five intercepts. The forest is also larger and slower in our local prediction test. My verdict is that its complexity is worthwhile when accuracy is the priority. Logistic Regression stays easier to explain. The split separates full artist credit strings, while collaborators can cross it, so the conclusion applies to this shared test set with that limitation.`;
s2.speakerNotes.text=`ABOUT 60 SECONDS\n${spoken2}\n\nNOT SPOKEN\n${source}F1 deltas RF minus LR: Electronic +0.069080, Rock +0.065514, Pop +0.065133, Latin +0.083254, Hip-hop/R&B +0.053508. The ±0.01 tie margin is descriptive, not a statistical equivalence test. The hip-hop/R&B precision comparison is 62.8% vs 40.4%, while recall is 12.86% vs 9.97%. 2,401 individual artist tokens cross the split and 5,072 test tracks have at least one token present in training. Do not claim all individual test artists are unseen.\n`;

const s3=await header('Some songs still favor logistic regression','Paired outcomes on the shared test set',C.forest,'BACKUP  /  PAIRED ERRORS');s3.hidden=true;
table(s3,[['Outcome','Songs'],['Both correct','4,547'],['Both wrong','2,657'],['Forest only correct','1,221'],['Logistic only correct','584']],72,250,475,300,[330,145],true,19);
txt(s3,'A forest-only success',625,248,560,40,20,C.mint,true);
txt(s3,'ZZ Top, “Sharp Dressed Man”',625,297,560,42,20,C.white,true);
txt(s3,'Actual: Rock\nLogistic: Electronic    Forest: Rock',625,344,560,74,17.5,C.white);
txt(s3,'A logistic-only success',625,451,560,38,20,C.mint,true);
txt(s3,'Eduardo Costa',625,495,560,34,20,C.white,true);
txt(s3,'“Esta Noite Foi Maravilhosa”\nActual: Latin    Logistic: Latin    Forest: Pop',625,537,560,75,17.5,C.white);
txt(s3,'Examples use the median true-class probability difference within each genre and paired-outcome group.',72,651,1136,34,12.75,'#B9D1C1');
s3.speakerNotes.text=`OPTIONAL BACKUP\nThe forest wins overall, but its predictions do not contain every logistic success. Both models get 4,547 songs right, and both miss 2,657. There are 1,805 songs where exactly one model is right. An additional set can have different wrong labels. The case examples are descriptive and do not establish why the models made these decisions.\n\n${source}Full examples and track IDs are in results/tables/representative_tracks.csv. The ZZ Top title in full is Sharp Dressed Man - 2008 Remaster. Example selection sorts by RF minus LR true-class probability within each genre/outcome and takes the middle row, with track ID as the tie-breaker.\n`;

let s4;
if(data.permutation_importance && new Set(data.permutation_importance.map(r=>r.model)).size===2){
 s4=await header('Feature importance on equal terms','Test macro F1 lost after shuffling one feature, averaged across five repeats',C.black,'BACKUP  /  INTERPRETATION');s4.hidden=true;
 const imp=data.permutation_importance.filter(r=>r.metric==='macro_f1');
 const features=[...new Set(imp.map(r=>r.feature))].sort((a,b)=>Math.max(...imp.filter(r=>r.feature===b).map(r=>r.mean_score_drop))-Math.max(...imp.filter(r=>r.feature===a).map(r=>r.mean_score_drop))).slice(0,6).reverse();
 s4.charts.add('bar',{position:{left:64,top:235,width:765,height:345},categories:features.map(x=>x.replace('duration_ms','Duration').replace(/^./,c=>c.toUpperCase())),
  series:['logistic_regression','random_forest'].map((m,i)=>({name:i?'Random Forest':'Logistic Regression',values:features.map(f=>Number(imp.find(r=>r.model===m&&r.feature===f).mean_score_drop.toPrecision(12))),fill:i?C.green:C.gray,valuesFormatCode:'0.000'})),
  barOptions:{direction:'bar',grouping:'clustered',gapWidth:50},hasLegend:true,
  legend:{position:'bottom',overlay:false,textStyle:{typeface:'Avenir Next',fontSize:'13pt',fill:C.white}},
  xAxis:{textStyle:{typeface:'Avenir Next',fontSize:'14pt',fill:C.white},line:{fill:'none',width:0}},
  yAxis:{min:0,max:.16,majorUnit:.04,numberFormatCode:'0.00',textStyle:{typeface:'Avenir Next',fontSize:'13pt',fill:C.gray},majorGridlines:{fill:'#34463B',width:.7}},
  dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.000',textStyle:{typeface:'Avenir Next',fontSize:'13pt',fill:C.white}},chartFill:C.black,plotAreaFill:C.black,chartLine:{fill:'none',width:0}});
 txt(s4,'Interpretation cost',890,254,318,45,21,C.mint,true);
 txt(s4,'70 weights\n5 intercepts',890,325,318,100,25,C.white,true);
 txt(s4,'300 trees\n1.78 million nodes',890,444,318,96,22,C.white,true);
 txt(s4,'A larger score drop means the model relies more on that feature.',72,603,1136,37,18,C.white);
 txt(s4,'Correlated features can share importance. Shuffling measures predictive reliance, with no causal claim.',72,654,1136,32,12.75,C.gray);
 s4.speakerNotes.text=`OPTIONAL BACKUP\nPermutation importance gives both models the same question. How much does test macro F1 drop if we shuffle one feature and preserve every other column? Five shuffles use seed 4241 for both verified models. The confirmed original forest and the LR refit reproduce every saved label and probability. Acousticness drops forest F1 by 0.1282 versus 0.0786 for LR. Instrumentalness drops it by 0.1016 versus 0.0511. Danceability hurts both by about 0.10. These are final-model explanations, never a basis for tuning or feature selection. Correlated inputs can share importance and shuffling can create unusual combinations, so these are not causal effects or a complete explanation of the accuracy gain.\n\n${source}The supplied original model is models/random_forest.joblib.gz.manifest.json (lossless parts; hash refers to original decompressed bytes), SHA-256 ${data.audit.fitted_rf_model.sha256}. Evaluation uses scikit-learn 1.9.1, matching its saved version. All 14 features, means, repeat standard deviations and accuracy/F1 scores are in results/tables/permutation_importance.csv. Model checks are in results/tables/model_verification.csv. Complexity measurements are in results/tables/complexity.csv. Serialized sizes are about 185.7 MB for RF and 2.3 KB for LR, uncompressed protocol 5. Median single-worker prediction times for 9,009 songs were about 0.256 s and 0.0021 s over five warmed-up batches. These are local timings. Original RF training wall time was not recorded.\n`;
} else if (data.feature_rankings_different_measures) {
 s4=await header('Audio feature rankings','Two importance definitions from the available model analyses',C.black,'BACKUP  /  INTERPRETATION');s4.hidden=true;
 const label=f=>f==='duration_ms'?'Duration':f.charAt(0).toUpperCase()+f.slice(1);
 const rows=data.feature_rankings_different_measures.slice(0,5).map(r=>[String(r.rank),label(r.lr_feature),label(r.rf_feature)]);
 table(s4,[['Rank','LR test permutation','RF training Gini'],...rows],72,250,1136,310,[80,528,528],true,20);
 txt(s4,'Both emphasize acousticness and danceability. The score definitions differ.',72,603,1136,45,18,C.white);
 txt(s4,'Common test permutation comparison still needs a verified fitted forest. RF ranks come from the teammate’s training-fitted clone.',72,654,1136,35,12.75,C.gray);
 s4.speakerNotes.text=`OPTIONAL BACKUP\nThe completed Random Forest notebook now supplies a native feature ranking. Its training-fitted clone ranks acousticness, danceability and instrumentalness highest by normalized Gini impurity reduction. Our verified Logistic Regression ranks danceability, acousticness and valence highest by the drop in test macro F1 after shuffling a feature. Both emphasize acousticness and danceability. These methods have different definitions and scales. This table compares names and within-model ranks qualitatively. It does not compare the numeric magnitudes or establish an exact feature-importance comparison for the saved forest. A common test-permutation comparison still requires a verified fitted forest.\n\n${source}Additional source: teammate_inputs/4_random_forest.ipynb, recorded output from the clone fitted on all training songs. Notebook SHA-256: ${data.teammate_input_audit.files['4_random_forest.ipynb'].sha256}. The notebook experiments were inspected without rerunning them. Extracted sources: results/tables/teammate_rf_gini_importance.csv and feature_rankings_different_measures.csv. The earlier model/requirements/dataset downloads remain excluded pending confirmation.\n`;
} else {
 s4=await header('What the comparison can establish','Verification and remaining dependencies',C.black,'BACKUP  /  METHODS');s4.hidden=true;
 table(s4,[['Check','Status'],['Same test IDs, labels, class order','Verified'],['Valid named probabilities','Verified'],['Held-out metrics and paired errors','Verified'],['Common feature importance','Needs verified forest refit'],['Individual artists fully separated','No, collaborations cross']],72,250,1136,315,[650,486],true,18);
 txt(s4,'The original Random Forest model artifact and training package versions would resolve the remaining refit dependency.',72,605,1136,65,18,C.white);
 s4.speakerNotes.text=`OPTIONAL BACKUP\nThe official test metrics use the saved prediction files and remain valid for this frozen split. The repository does not include the fitted forest or training package versions. Refit results must match the saved labels and probabilities before comparing feature importance or reporting those refits as the official models. The analysis code enforces this rule.\n\n${source}`;
}

await fs.mkdir(path.join(base,'draft_render'),{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1.5});
 await fs.writeFile(path.join(base,`draft_render/slide-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await p.slides.items[i].export({format:'layout'});
 await fs.writeFile(path.join(base,`draft_render/slide-${i+1}.json`),await layout.text());
}
await (await PresentationFile.exportPptx(p)).save(path.join(base,'section-draft.pptx'));
await fs.writeFile(path.join(base,'spoken_notes.json'),JSON.stringify({slide1:spoken1,slide2:spoken2},null,2));
console.log('Built 2 main slides and 2 hidden backups.');
