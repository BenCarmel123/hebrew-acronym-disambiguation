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
  const summary=ui.renderCase({id:'old',inherited_from:{schema:'old'},answers:[{model:'Visible after summary',text:'answer',label:'fits',comparison:'agreement'}]});
  assert.ok(summary.includes('אין כאן תיוג עצמאי חדש'));
  assert.ok(summary.includes('תואם לניקוד האוטומטי'));
  assert.ok(!summary.includes('agreement'));
});
