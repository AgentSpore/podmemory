const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../src/podmemory/static/index.html'), 'utf8');
const source = html.slice(html.indexOf("document.getElementById('importBtn').onclick"), html.indexOf('// ─── SM-2 ───'));
function importFile(payload) {
  const original = [{id:'existing', title:'Existing', flashcards:[]}];
  const state = {library:original, srData:{'existing:0':{ef:2.5, interval:1, repetitions:1, nextReview:10}}};
  const storage = new Map([['pm_library',JSON.stringify(state.library)], ['pm_sr',JSON.stringify(state.srData)]]);
  const before = [...storage];
  const elements = {importBtn:{},importFile:{}};
  const messages = [];
  const context = vm.createContext({...state, document:{getElementById:id=>elements[id]},
    localStorage:{setItem:(key,value)=>storage.set(key,value)},
    FileReader:class {readAsText(file) {this.onload({target:{result:file}});}},
    renderLibrary:()=>{}, updateQuizBadge:()=>{}, toast:(...args)=>messages.push(args)});
  vm.runInContext(source,context);
  elements.importFile.onchange({target:{files:[payload],value:'file'}});
  return {context,storage,before,messages,state};
}
const valid = {id:'new',title:'Synthetic',flashcards:[{q:'Q',a:'A'}]};
const invalid = [{library:{}},{library:[null]},{library:[{id:'a',title:7}]},
  {library:[],srData:[]},null,{},[],{library:null},{srData:null},
  ...[{flashcards:{}},{flashcards:[null]},{flashcards:[{q:'Q',a:4}]},
      {tags:[7]},{timestamps:[{time:7,label:'x'}]},
      {vocabulary:[{term:'x'}]},{quotes:{}},{url:7},{id:"x'bad"}]
    .map(fields=>({library:[{...valid,...fields}]})),
  {library:[valid,valid]},
  ...[null,[],{ef:2.5,interval:1,repetitions:0,nextReview:-1},
      {ef:2.5,interval:1,repetitions:0,nextReview:'tomorrow'}]
    .map(card=>({library:[],srData:{'new:0':card}}))];
for (const data of invalid) {
  test('reject invalid backup without writes: '+JSON.stringify(data),()=>{
    const result = importFile(JSON.stringify(data));
    assert.deepEqual([...result.storage],result.before);
    assert.equal(result.context.library,result.state.library);
    assert.equal(result.context.srData,result.state.srData);
    assert.equal(result.messages.at(-1)[1],'error');
  });
}
test('accept own valid backup',()=>{
  const data={library:[{id:'new',title:'Synthetic',flashcards:[{q:'Q',a:'A'}],tags:[]}],srData:{}};
  const result=importFile(JSON.stringify(data));
  assert.equal(result.storage.get('pm_library'),JSON.stringify(data.library));
  assert.equal(result.storage.get('pm_sr'),'{}');
  assert.equal(result.messages.at(-1)[1],'success');
});

test('reject malformed JSON without writes',()=>{
  const result=importFile('{');
  assert.deepEqual([...result.storage],result.before);
  assert.equal(result.messages.at(-1)[1],'error');
});
test('partial library backup preserves review state and unknown fields',()=>{
  const data={library:[{...valid,custom:'kept'}]};
  const result=importFile(JSON.stringify(data));
  assert.equal(result.storage.get('pm_library'),JSON.stringify(data.library));
  assert.equal(result.storage.get('pm_sr'),result.before[1][1]);
});
test('partial schedule backup preserves library',()=>{
  const result=importFile(JSON.stringify({srData:{}}));
  assert.equal(result.storage.get('pm_library'),result.before[0][1]);
  assert.equal(result.storage.get('pm_sr'),'{}');
});

test('round trip through actual export and import handlers',()=>{
  const library=[{...valid,tldr:'Summary',tags:['tag'],key_insights:['Insight'],
    quotes:['Quote'],action_items:['Action'],timestamps:[{time:'0:01',label:'Start'}],
    vocabulary:[{term:'Word',definition:'Meaning'}],url:'https://example.com',
    analyzed_at:'2026-11-02T00:00:00Z',difficulty:'beginner',model_used:'synthetic'}];
  const srData={'new:0':{ef:2.5,interval:1,repetitions:1,nextReview:100}};
  let exported;
  const button={};
  const context=vm.createContext({library,srData,Date,
    Blob:class {constructor(parts){this.text=parts.join('');}},
    URL:{createObjectURL:blob=>{exported=blob.text;return 'blob:synthetic';},revokeObjectURL:()=>{}},
    document:{getElementById:()=>button,createElement:()=>({click(){}})},
    toast:()=>{},t:key=>key});
  const start=html.indexOf("document.getElementById('exportBtn').onclick");
  const end=html.indexOf("document.getElementById('importBtn').onclick");
  vm.runInContext(html.slice(start,end),context);
  button.onclick();
  const result=importFile(exported);
  assert.equal(result.storage.get('pm_library'),JSON.stringify(library));
  assert.equal(result.storage.get('pm_sr'),JSON.stringify(srData));
  assert.equal(result.messages.at(-1)[1],'success');
});
