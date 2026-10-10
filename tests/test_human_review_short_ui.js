'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const ui=require('../src/hebrew_acronyms/human_review_web/short.js');
const fixture={id:'item-1',sentence:'משפט עם אב״ג.',acronym:'אב״ג',gold:'ייחוס',target_spans:[{start:8,end:12}],answers:[{id:'opaque-1',code:'א',text:'פירוש ראשון',model:'HIDDEN_MODEL_A',auto_score:true,comparison:'HIDDEN_MATCH'},{id:'opaque-2',code:'ב',text:'פירוש שני',model:'HIDDEN_MODEL_B',auto_score:false}],selection_reason:'HIDDEN_REASON'};
test('main view whitelists neutral response fields and hides identities, scores and reasons',()=>{
  const html=ui.renderMain(fixture,{});
  for(const secret of ['HIDDEN_MODEL_A','HIDDEN_MODEL_B','HIDDEN_MATCH','HIDDEN_REASON','auto_score','comparison'])assert.ok(!html.includes(secret),secret);
  assert.ok(html.includes('תשובה א'));assert.ok(html.includes('תשובה ב'));
  assert.ok(html.includes('פירוש ראשון'));assert.ok(html.includes('פירוש שני'));
});
test('two responses have exactly the three requested semantic options each',()=>{
  const html=ui.renderMain(fixture,{});
  assert.equal((html.match(/type="radio"/g)||[]).length,6);
  assert.deepEqual(ui.LABELS,{fits:'מתאימה להקשר',not_fits:'לא מתאימה',unsure:'לא בטוח'});
  for(const value of Object.keys(ui.LABELS))assert.equal((html.match(new RegExp('value="'+value+'"','g'))||[]).length,2);
});
test('reused completed record is skipped at startup while partial progress is retained',()=>{
  const state={queue:['old','partial','new'],records:{old:{completion:{status:'complete'}},partial:{completion:{status:'partial'}}}};
  assert.equal(ui.firstIncomplete(state),'partial');
  state.records.partial.completion.status='complete';assert.equal(ui.firstIncomplete(state),'new');
  state.records.new={completion:{status:'complete'}};assert.equal(ui.firstIncomplete(state),undefined);
});
test('unsure is a valid marked response; blank and unknown labels do not complete it',()=>{
  assert.deepEqual(ui.progressFor(fixture,{judgments:{'opaque-1':{label:'unsure'},'opaque-2':{label:''}}}),{judged:1,total:2});
  assert.deepEqual(ui.progressFor(fixture,{judgments:{'opaque-1':{label:'not_fits'},'opaque-2':{label:'fits'}}}),{judged:2,total:2});
  assert.deepEqual(ui.progressFor(fixture,{judgments:{'opaque-1':{label:'legacy_partial'}}}),{judged:0,total:2});
});
test('saved choices and optional fields render without pre-filling untouched decisions',()=>{
  const html=ui.renderMain(fixture,{judgments:{'opaque-1':{label:'unsure'}},suspect:true,note:'הערה'});
  assert.equal((html.match(/value="unsure" checked/g)||[]).length,1);
  assert.equal((html.match(/type="radio"[^>]*checked/g)||[]).length,1);
  assert.ok(html.includes('id="suspect" type="checkbox" checked'));
  assert.ok(html.includes('הערה'));
});
test('source text, annotations and opaque IDs are escaped as data',()=>{
  const malicious={...fixture,sentence:'<script>bad()</script>',gold:'<img onerror="bad()">',target_spans:[],answers:[{id:'x" onclick="bad()',code:'א',text:'<svg onload=bad()>'}]};
  const html=ui.renderMain(malicious,{note:'</textarea><script>bad()</script>'});
  assert.ok(!html.includes('<script>'));assert.ok(!html.includes('<svg'));assert.ok(!html.includes('<img'));
  assert.ok(html.includes('&lt;script&gt;'));assert.ok(html.includes('x&quot; onclick=&quot;bad()'));
});
test('reused work has a notice but no new annotation requirement',()=>{
  const html=ui.renderMain({...fixture,prior_reused:true},{judgments:{'opaque-1':'fits','opaque-2':'not_fits'},completion:{status:'complete'}});
  assert.ok(html.includes('אין צורך לבדוק שוב תשובה שכבר נשפטה'));
  assert.equal((html.match(/type="radio"[^>]*checked/g)||[]).length,2);
  assert.ok(!html.includes('interpretation'));assert.ok(!html.includes('annotator'));
});
test('main shell has exact short guidance and one details entry, without accordions',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../src/hebrew_acronyms/human_review_web/short.html'),'utf8');
  assert.ok(html.includes('שפוט משמעות בהקשר, לא התאמה מילולית לייחוס. אם חסר בסיס להחלטה — לא בטוח.'));
  assert.ok(html.includes('נתקעת? סמן לא בטוח והמשך. אין צורך לחפש מידע חיצוני.'));
  assert.equal((html.match(/id="detailsButton"/g)||[]).length,1);
  assert.ok(!html.includes('<details'));
});

test('partial inherited work asks only for missing judgments and does not claim completion',()=>{
  const html=ui.renderMain({...fixture,prior_reused:true},{judgments:{'opaque-1':{label:'fits'}},completion:{status:'partial'}});
  assert.ok(html.includes('משלימים רק שיפוטים חסרים'));
  assert.equal((html.match(/type="radio"[^>]*checked/g)||[]).length,1);
});
test('legacy metadata never enters main view and summary marks inherited origin explicitly',()=>{
  const record={legacy_record:{answers:{secret:{model:'HIDDEN_LEGACY_MODEL'}}},inherited_from:{schema:'old'},judgments:{}};
  assert.ok(!ui.renderMain(fixture,record).includes('HIDDEN_LEGACY_MODEL'));
  const summary=ui.renderCase({id:'old',inherited_from:{schema:'old'},answers:[{model:'Visible after summary',text:'answer',label:'fits',comparison:'agreement'}]},true);
  assert.ok(summary.includes('אין להציגם כתיוג עצמאי חדש'));
  assert.ok(summary.includes('תואם לניקוד האוטומטי'));
  assert.ok(!summary.includes('agreement'));
});

test('six optional tag chips render per answer without preselection or judgment inference',()=>{
  const html=ui.renderMain(fixture,{});
  for(const id of ['gibberish','inflection','punctuation','spelling','equivalent','extra_text']){
    assert.equal((html.match(new RegExp('value="'+id+'"','g'))||[]).length,2);
    assert.ok(!html.includes('value="'+id+'" checked'));
  }
  assert.ok(fs.readFileSync(path.join(__dirname,'../src/hebrew_acronyms/human_review_web/short.html'),'utf8').includes('סמן תופעה בולטת אם יש; אין צורך לחפש בכוח או להסביר כל החלטה.'));
  const tagged={judgments:{'opaque-1':{label:'',tags:['gibberish','extra_text']}}};
  assert.deepEqual(ui.progressFor(fixture,tagged),{judged:0,total:2});
  const saved=ui.renderMain(fixture,tagged);
  assert.ok(saved.includes('value="gibberish" checked'));
  assert.equal((saved.match(/type="radio"[^>]*checked/g)||[]).length,0);
});
test('tags remain tied to opaque response IDs after response order changes',()=>{
  const r={judgments:{'opaque-1':{label:'fits',tags:['spelling']},'opaque-2':{label:'not_fits',tags:['extra_text']}}};
  const reversed={...fixture,answers:[{...fixture.answers[1],code:'א'},{...fixture.answers[0],code:'ב'}]};
  const html=ui.renderMain(reversed,r);
  const cards=html.split('<article class="answer-row"');
  assert.ok(cards[1].includes('value="extra_text" checked'));
  assert.ok(cards[1].includes('value="not_fits" checked'));
  assert.ok(cards[2].includes('value="spelling" checked'));
  assert.ok(cards[2].includes('value="fits" checked'));
});
test('empty tags mean not marked and do not assert clean output',()=>{
  assert.equal(ui.tagsText({tags:[],tag_status:'not_marked'}),'לא סומן');
  const html=ui.renderMain(fixture,{});
  assert.ok(fs.readFileSync(path.join(__dirname,'../src/hebrew_acronyms/human_review_web/short.html'),'utf8').includes('ולא אישור שאין תופעות'));
});
test('masked summary strips models, scores, sources, reasons and comparison cues even from full input',()=>{
  const secretCase={...fixture,source:'SECRET_SOURCE',reason:'SECRET_REASON',legacy_record:{secret:'SECRET_LEGACY'},answers:[{id:'opaque',original_answer_id:'SECRET_ORIGINAL',code:'א',model:'SECRET_MODEL',text:'מקור טקסט',auto_score:'SECRET_SCORE',label:'fits',tags:['punctuation'],comparison:'disagreement',label_phase:'before_reveal'}]};
  const summary={masked:true,counts:{},cases:[secretCase],disagreements:[secretCase],accepted_human_rejected_auto:[secretCase],composition:{secret:'SECRET_COMPOSITION'},scoring_rules:'SECRET_RULES',recommendations:['SECRET_RECOMMENDATION']};
  const html=ui.renderSummaryContent(summary,true);
  for(const secret of ['SECRET_SOURCE','SECRET_REASON','SECRET_LEGACY','SECRET_ORIGINAL','SECRET_MODEL','SECRET_SCORE','SECRET_COMPOSITION','SECRET_RULES','SECRET_RECOMMENDATION','פער מול הניקוד האוטומטי'])assert.ok(!html.includes(secret),secret);
  assert.ok(html.includes('רווחים / מקפים / גרשיים'));
  assert.ok(html.includes('בפרוטוקול החדש לפני חשיפה'));
  assert.ok(html.includes('מקור טקסט'));
});
test('details always allow only candidates and never legacy or response metadata',()=>{
  const html=ui.renderMaskedDetails({candidates:['פירוש'],source:'SECRET_SOURCE',source_metadata:{model:'SECRET_MODEL'},selection_reason:'SECRET_REASON',legacy_record:{model:'SECRET_LEGACY'},answers:[{model:'SECRET_MODEL',auto_score:'SECRET_SCORE'}]});
  assert.ok(html.includes('פירוש'));
  for(const secret of ['SECRET_SOURCE','SECRET_MODEL','SECRET_REASON','SECRET_LEGACY','SECRET_SCORE'])assert.ok(!html.includes(secret));
});
test('full result rendering requires explicit permission and supports both discrepancy directions',()=>{
  const value={id:'sentence',source:'SOURCE_FULL',gold:'ייחוס',note:'הערה',answers:[{original_answer_id:'ORIGINAL_FULL',model:'MODEL_FULL',text:'טקסט מלא',label:'fits',auto_score:false,tags:['spelling'],label_phase:'prior_protocol',tags_phase:'after_reveal'}]};
  const summary={masked:false,counts:{},cases:[value],accepted_human_rejected_auto:[value],rejected_human_accepted_auto:[],tag_counts:{spelling:1}};
  assert.ok(!ui.renderSummaryContent(summary).includes('MODEL_FULL'));
  const full=ui.renderSummaryContent(summary,true);
  for(const expected of ['MODEL_FULL','ORIGINAL_FULL','SOURCE_FULL','נפסלה אוטומטית והאדם קיבל','התקבלה אוטומטית והאדם דחה','לאחר חשיפה מתועדת','הבדל כתיב'])assert.ok(full.includes(expected),expected);
});

test('previous notes are preserved without revealing them during labeling',()=>{
  const r={has_previous_note:true,previous_note:'SECRET_OLD_NOTE',judgments:{}};
  const main=ui.renderMain(fixture,r);
  assert.ok(main.includes('הערה קודמת נשמרה ותוצג לאחר החשיפה'));
  assert.ok(!main.includes('SECRET_OLD_NOTE'));
  const c={...fixture,previous_note:'SECRET_OLD_NOTE'};
  assert.ok(!ui.renderCase(c).includes('SECRET_OLD_NOTE'));
  assert.ok(ui.renderCase(c,true).includes('SECRET_OLD_NOTE'));
  assert.ok(ui.fullCasesTable([c]).includes('SECRET_OLD_NOTE'));
});

test('tag checkbox accessible names stay stable before and after selection',()=>{
  const plain=ui.renderMainAnswer(fixture.answers[0],'',[]);
  const checked=ui.renderMainAnswer(fixture.answers[0],'',['spelling']);
  for(const [id,label] of Object.entries(ui.TAGS)){
    const expected='type="checkbox" aria-label="'+ui.esc(label)+'" value="'+id+'"';
    assert.ok(plain.includes(expected));
    assert.ok(checked.includes(expected));
  }
  assert.ok(checked.includes('aria-label="הבדל כתיב, למשל י׳/ו׳" value="spelling" checked'));
});

test('an empty human label is unjudged rather than unknown exposure',()=>{
  const answer={code:'א',text:'טקסט',label:'',label_phase:null,tags:[]};
  assert.equal(ui.judgmentPhaseText(answer),'טרם נשפט');
  const c={id:'one',answers:[answer]};
  for(const html of [ui.renderCase(c),ui.fullCasesTable([c])]){
    assert.ok(html.includes('טרם נשפט'));
    assert.ok(!html.includes('מצב החשיפה אינו ידוע'));
  }
  assert.equal(ui.judgmentPhaseText({...answer,label:'fits',label_phase:'unknown'}),'מצב החשיפה אינו ידוע');
});

test('actual selected and reviewed composition is rendered only after explicit reveal',()=>{
  const composition={selected:{generation_fail_selection_pass:8,both_fail:6,both_generation_pass:6},reviewed:{generation_fail_selection_pass:3},completed:{generation_fail_selection_pass:2},sources_selected:{SOURCE_COMPOSITION:20},sources_reviewed:{SOURCE_COMPOSITION:3},adjustments:[]};
  const summary={masked:false,counts:{},cases:[],composition};
  const full=ui.renderSummaryContent(summary,true);
  assert.ok(full.includes('הרכב הרשימה והבדיקה בפועל'));
  assert.ok(full.includes('<td>8</td><td>3</td><td>2</td>'));
  assert.ok(full.includes('SOURCE_COMPOSITION'));
  for(const masked of [ui.renderSummaryContent(summary),ui.renderSummaryContent({...summary,masked:true},true)]){
    assert.ok(!masked.includes('SOURCE_COMPOSITION'));
    assert.ok(!masked.includes('כשל ביצירה והצלחה בבחירה של אותו מודל'));
    assert.ok(!masked.includes('הרכב הרשימה והבדיקה בפועל'));
  }
});

test('continuation starts with backend pending queue rather than old item completion',()=>{
  const state={queues:{all:['remaining'],unsure:['uncertain'],foreign:['remaining'],filtered:['match'],suspicions:['suspect']},queue:['remaining'],records:{remaining:{completion:{status:'complete'}}}};
  assert.equal(ui.firstIncomplete(state),'remaining');
  assert.deepEqual(ui.queueIds(state,'unsure'),['uncertain']);
  assert.deepEqual(ui.queueIds(state,'all'),['remaining']);
});
test('next navigation handles queue shrink after autosave and keeps earlier skips reachable',()=>{
  assert.equal(ui.nextQueueId(['a','b','c'],'b',['a','c']),'c');
  assert.equal(ui.nextQueueId(['a','b','c'],'c',['a']),'a');
  assert.equal(ui.nextQueueId(['a','b'],'b',['b']),undefined);
});
test('single remaining or duplicate-collapsed answer requires only one judgment',()=>{
  const i={...fixture,answers:[{...fixture.answers[0],review_status:'pending',occurrence_count:2}]};
  const html=ui.renderMain(i,{});
  assert.equal((html.match(/type="radio"/g)||[]).length,3);
  assert.ok(html.includes('שיפוט אחד חל על 2 מופעים זהים באותו משפט'));
  assert.deepEqual(ui.progressFor(i,{}),{judged:0,total:1});
  assert.ok(!html.includes('HIDDEN_MODEL'));
});
test('mechanical answers have read-only cards with restoration but no judgment controls',()=>{
  for(const status of ['exacttrim','technical','missing']){
    const i={...fixture,answers:[{...fixture.answers[0],review_status:status}]};
    const html=ui.renderMain(i,{},'filtered');
    assert.ok(!html.includes('type="radio"'));
    assert.ok(!html.includes('data-tags-answer'));
    assert.ok(html.includes('data-restore-answer="opaque-1"'));
    assert.ok(!html.includes('HIDDEN_MODEL_A'));
  }
});
test('partition keeps duplicate savings and foreign letters separate from primary groups',()=>{
  const counts={source_answers:790,human_answers:40,exacttrim_answers:300,technical_answers:50,missing_answers:10,pending_answers:390,pending_decisions:360,duplicate_savings:30,foreign_pending_answers:20};
  const html=ui.renderPartition(counts);
  assert.ok(html.includes('<th>סכום הקבוצות הראשיות</th><td>790</td>'));
  assert.ok(html.includes('30 הצגות נוספות'));
  assert.ok(html.includes('20 מהממתינות'));
  assert.ok(html.includes('<strong>360</strong> הכרעות'));
});
test('historical reveal does not automatically unmask continuation summary',()=>{
  const state={revealed:true,queues:{all:[]}};
  assert.equal(ui.canShowFull(state,{masked:false}),false);
  assert.equal(ui.canShowFull(state,{masked:false},true),true);
  assert.equal(ui.canShowFull(state,{masked:true},true),false);
});
test('continuation shell has all queues, secondary filtered access and no twenty-item hour promise',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../src/hebrew_acronyms/human_review_web/short.html'),'utf8');
  for(const v of ['all','foreign','unsure','suspicions'])assert.ok(html.includes('value="'+v+'"'));
  assert.ok(html.includes('עיון במסוננות והחזרה לבדיקה'));
  assert.ok(!html.includes('עד שעה'));
  assert.ok(!html.includes('20 משפטים'));
});

test('Enter queues SaveNext during autosave and persists edits made while the earlier request is pending',async()=>{
  const vm=require('node:vm');
  const source=fs.readFileSync(path.join(__dirname,'../src/hebrew_acronyms/human_review_web/short.js'),'utf8');
  const elements=new Map();
  function element(id){
    if(!elements.has(id))elements.set(id,{value:'',checked:false,hidden:false,disabled:false,inert:false,innerHTML:'',textContent:'',listeners:{},addEventListener(type,fn){this.listeners[type]=fn;},querySelectorAll(){return [];}});
    return elements.get(id);
  }
  let label='',tags=[],debounce,serverRevision=0;
  const main=element('main'),qualityInput=element('qualityInput'),tagInput=element('tagInput');
  element('itemView').querySelectorAll=selector=>selector==='input,textarea'?[qualityInput,tagInput]:[];
  const group={dataset:{answer:'opaque-1'},querySelector(){return label?{value:label}:null;}};
  const tagGroup={dataset:{tagsAnswer:'opaque-1'},querySelectorAll(){return tags.map(value=>({value}));}};
  const document={getElementById:element,querySelector:()=>main,querySelectorAll:selector=>selector==='[data-answer]'?[group]:selector==='[data-tags-answer]'?[tagGroup]:[]};
  const state=()=>({protocol_version:'qualitative-generation-v3',revision:serverRevision,queue:serverRevision>1?['second']:['first','second'],queues:{all:serverRevision>1?['second']:['first','second'],foreign:[],unsure:[],suspicions:[],filtered:[]},records:{},counts:{pending_answers:2,pending_decisions:2,human_answers:0},reviewer:'QA',revealed:false});
  const opened=[],saves=[],waiting=[],windowEvents={};
  const fetch=async(url,options)=>{
    const payload=options?.body?JSON.parse(options.body):null;
    if(url==='/api/short/state')return {ok:true,json:async()=>state()};
    if(url==='/api/session')return {ok:true,json:async()=>({qa:true})};
    if(url==='/api/short/open'){
      opened.push(payload.item_id);serverRevision++;
      return {ok:true,json:async()=>({state:state(),item:{...fixture,id:payload.item_id,answers:[{id:'opaque-1',code:'א',text:'פירוש',review_status:'pending'}]}})};
    }
    if(url==='/api/short/save'){
      saves.push(payload);
      return new Promise(resolve=>waiting.push(()=>{serverRevision++;resolve({ok:true,json:async()=>state()});}));
    }
    throw new Error('Unexpected request '+url);
  };
  vm.runInNewContext(source,{document,fetch,setInterval:()=>0,setTimeout:fn=>{debounce=fn;return 1;},clearTimeout:()=>{},window:{addEventListener(type,listener){windowEvents[type]=listener;},scrollTo(){}},console});
  async function until(predicate){for(let n=0;n<30&&!predicate();n++)await new Promise(resolve=>setImmediate(resolve));assert.ok(predicate(),'Expected asynchronous milestone: '+element('saveStatus').textContent);}
  await until(()=>opened.length===1&&Boolean(qualityInput.listeners.input)&&!main.inert);
  label='fits';qualityInput.listeners.input();debounce();
  await until(()=>saves.length===1);
  assert.equal(main.inert,false,'Background autosave must leave navigation clickable');
  tags=['spelling'];tagInput.listeners.input();
  windowEvents.keydown({key:'Enter',target:{tagName:'TEXTAREA'},preventDefault(){throw new Error('Typing shortcut must be ignored');}});
  assert.equal(main.inert,false,'Enter inside a note must not navigate');
  let prevented=false;windowEvents.keydown({key:'Enter',target:{tagName:'ARTICLE'},preventDefault(){prevented=true;}});
  assert.equal(prevented,true);
  assert.equal(main.inert,true,'Explicit navigation locks the form while awaiting save');
  waiting.shift()();
  await until(()=>saves.length===2);
  assert.deepEqual(saves[0].tags['opaque-1'],[]);
  assert.deepEqual(saves[1].tags['opaque-1'],['spelling']);
  assert.equal(opened.length,1,'Do not navigate before the latest edit is saved');
  waiting.shift()();
  await until(()=>opened.length===2&&!main.inert);
  assert.deepEqual(opened,['first','second']);
  assert.equal(saves[1].judgments['opaque-1'],'fits');
});

test('zero pending continuation answers reports completion despite mechanical exclusions',()=>{
  const html=ui.renderSummaryContent({masked:true,counts:{source_answers:790,pending_answers:0,pending_decisions:0,human_answers:500,exacttrim_answers:280,technical_answers:10,missing_answers:0,complete_items:200,total_items:395},incomplete_ids:[],cases:[]});
  assert.ok(html.includes('אין תשובות שממתינות לבדיקה; שיפוטי לא בטוח נשארים בתור החזרה.'));
  assert.ok(!html.includes('אפשר להמשיך בפריטים שטרם הושלמו.'));
});

test('numeric shortcuts bind only to the active row and do not infer a tag',()=>{
  assert.deepEqual(ui.shortcutIntent({key:'1',target:{tagName:'ARTICLE'}},'opaque-B'),{type:'label',answerId:'opaque-B',label:'fits'});
  assert.deepEqual(ui.shortcutIntent({key:'2',target:{tagName:'INPUT',type:'radio'}},'opaque-A'),{type:'label',answerId:'opaque-A',label:'not_fits'});
  assert.deepEqual(ui.shortcutIntent({key:'3',target:{}},'opaque-A'),{type:'label',answerId:'opaque-A',label:'unsure'});
  assert.equal(ui.shortcutIntent({key:'1',target:{}},null),null);
});
test('keyboard shortcuts ignore typing fields, repeat and modifier keys',()=>{
  for(const target of [{tagName:'TEXTAREA'},{tagName:'SELECT'},{tagName:'INPUT',type:'text'},{tagName:'INPUT',type:'number'},{isContentEditable:true}]){
    for(const key of ['1','2','3','Enter'])assert.equal(ui.shortcutIntent({key,target},'answer'),null);
  }
  for(const flag of ['repeat','ctrlKey','altKey','metaKey','shiftKey'])assert.equal(ui.shortcutIntent({key:'1',[flag]:true,target:{}},'answer'),null);
  assert.equal(ui.shortcutIntent({key:'Enter',target:{tagName:'BUTTON'}},'answer'),null);
  assert.deepEqual(ui.shortcutIntent({key:'Enter',target:{tagName:'ARTICLE'}},'answer'),{type:'next'});
});
test('active row advances to another unjudged response without advancing item',()=>{
  assert.equal(ui.nextActiveAnswer(['a','b'],{a:'fits'},'a'),'b');
  assert.equal(ui.nextActiveAnswer(['a','b'],{a:'fits',b:'unsure'},'b'),'b');
  assert.equal(ui.nextActiveAnswer(['a','b'],{b:'not_fits'},'b'),'a');
  assert.equal(ui.nextActiveAnswer(['a'],{a:'fits'},'a'),'a');
});
test('compact rows share sentence context and preserve original response whitespace',()=>{
  const i={...fixture,answers:[{id:'a',code:'א',text:'  מענה\nמקורי  ',occurrence_count:2},{id:'b',code:'ב',text:'אחר'}]};
  const html=ui.renderMain(i,{});
  assert.equal((html.match(/class="sentence"/g)||[]).length,1);
  assert.equal((html.match(/class="answer-row"/g)||[]).length,2);
  assert.ok(html.includes('  מענה\nמקורי  '));
  assert.ok(!html.includes('tag-help'));
  assert.ok(html.includes('ייחוס (עשוי להיות שגוי)'));
});
