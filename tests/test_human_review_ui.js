'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const logic = require('../src/hebrew_acronyms/human_review_web/review_logic.js');
const item = {id:'one',source:'fixture',acronym:'ABC',findings:['failure'],answers:[{id:'a',system_id:'A',mechanical_flags:[]},{id:'b',system_id:'B',mechanical_flags:['failure']}]};
const record = {draft:{item_problems:['suspect_gold'],answers:{a:{quality:'correct',system_id:'A',disagrees_auto:'no'},b:{quality:'wrong',system_id:'B',disagrees_auto:'yes'}}},completion:{status:'complete'}};
test('wrong and disagreement are scoped to the selected answer system',()=>{
  for(const finding of ['wrong','disagreement']){
    assert.equal(logic.matches(item,record,{system:'A',finding}),false);
    assert.equal(logic.matches(item,record,{system:'B',finding}),true);
    assert.equal(logic.matches(item,record,{finding}),true);
  }
});
test('partial is independent from undecidable and obeys system selection',()=>{
  const r={draft:{answers:{a:{quality:'partial'},b:{quality:'undecidable'}}}};
  assert.equal(logic.matches(item,r,{system:'A',finding:'partial'}),true);
  assert.equal(logic.matches(item,r,{system:'A',finding:'undecidable'}),false);
  assert.equal(logic.matches(item,r,{system:'B',finding:'partial'}),false);
  assert.equal(logic.matches(item,r,{system:'B',finding:'undecidable'}),true);
});
test('item-level findings remain available for every system',()=>{
  assert.equal(logic.matches(item,record,{system:'A',finding:'suspect_gold'}),true);
  assert.equal(logic.matches(item,record,{system:'B',finding:'suspect_gold'}),true);
});
test('answer mechanical findings do not leak from another system',()=>{
  assert.equal(logic.matches(item,record,{system:'A',finding:'mechanical'}),false);
  assert.equal(logic.matches(item,record,{system:'B',finding:'mechanical'}),true);
});
test('summary uses supplied filtered items and system answer denominator',()=>{
  assert.deepEqual(logic.summary([item],{one:record},'A'),{items:1,complete:1,partial:0,draft:0,judged:1,possible:1,qualities:{correct:1}});
  assert.deepEqual(logic.summary([],{one:record},'B'),{items:0,complete:0,partial:0,draft:0,judged:0,possible:0,qualities:{}});
});
test('name-only and revisit-only historical review are not complete',()=>{
  const r={draft:{annotator:'test',revisit:true},reviewed:{annotator:'test'},completion:{status:'complete'}};
  assert.equal(logic.completion(item,r).status,'draft');
  assert.equal(logic.matches(item,r,{status:'complete'}),false);
  assert.equal(logic.completion(item,r).judged_answers,0);
});
test('an edited former completion uses the current draft, not historical review',()=>{
  const r={reviewed:record.draft,draft:{item_problems:['valid'],answers:{a:{quality:'correct'},b:{quality:''}}},completion:{status:'partial'}};
  assert.equal(logic.completion(item,r).status,'partial');
  assert.equal(logic.completion(item,r).judged_answers,1);
  assert.equal(logic.summary([item],{one:r}).complete,0);
});
test('legacy combined labels require revisit and do not count as current judgments',()=>{
  const r={draft:{item_problems:['valid'],answers:{a:{quality:'legacy_partial'},b:{quality:'undecidable'}}},completion:{status:'partial'}};
  assert.equal(logic.completion(item,r).judged_answers,1);
  assert.equal(logic.summary([item],{one:r}).judged,1);
  assert.equal(logic.matches(item,r,{finding:'partial'}),false);
  assert.equal(logic.matches(item,r,{finding:'legacy_partial'}),true);
});
