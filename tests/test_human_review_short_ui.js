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
  assert.ok(html.includes('אין צורך לבדוק אותו שוב'));
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
  assert.ok(html.includes('יש להשלים רק שיפוטים חסרים'));
  assert.ok(!html.includes('אין צורך לבדוק אותו שוב'));
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
  assert.ok(html.includes('סמן תופעה בולטת אם יש; אין צורך לחפש בכוח או להסביר כל החלטה.'));
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
  const cards=html.split('<article class="card"><h3>');
  assert.ok(cards[1].includes('value="extra_text" checked'));
  assert.ok(cards[1].includes('value="not_fits" checked'));
  assert.ok(cards[2].includes('value="spelling" checked'));
  assert.ok(cards[2].includes('value="fits" checked'));
});
test('empty tags mean not marked and do not assert clean output',()=>{
  assert.equal(ui.tagsText({tags:[],tag_status:'not_marked'}),'לא סומן');
  const html=ui.renderMain(fixture,{});
  assert.ok(html.includes('ולא אישור שאין תופעות'));
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
