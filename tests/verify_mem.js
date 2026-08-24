/*
 * LLM 预训练集群仿真器 · 回归测试
 * 用法：node tests/verify_mem.js（在项目根目录运行）
 * 用 DOM 桩执行 index.html 内的脚本，校验显存公式、单位换算、SwapOptimizer、
 * Layer Input、重算/预设/导入导出等全部逻辑。
 */
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error('no <script> block found');
let script = m[1];

/* 1. 纯语法检查 */
new Function(script);
console.log('[OK] JS syntax');

/* 2. 所有 $("...") 引用的 id 必须在 HTML 中存在 */
const idsUsed = [...script.matchAll(/\$\("([^"]+)"\)/g)].map(x => x[1]);
const missing = idsUsed.filter(id => !html.includes('id="' + id + '"'));
if (missing.length) { console.error('[FAIL] missing HTML ids:', missing); process.exit(1); }
console.log('[OK] all', idsUsed.length, 'JS-referenced ids exist in HTML');

/* 3. DOM 桩 */
function makeEl(id) {
  return { id, value: '', checked: false, textContent: '', innerHTML: '', className: '',
           options: [], selectedIndex: 0, addEventListener() {}, style: {} };
}
const els = {};
class CList {
  constructor(){ this.s = new Set(); }
  add(c){ this.s.add(c); } remove(c){ this.s.delete(c); } contains(c){ return this.s.has(c); }
  toggle(c, force){ if (force) this.s.add(c); else this.s.delete(c); }
}
global.document = {
  getElementById(id) {
    if (!els[id]) { const e = makeEl(id); e.classList = new CList(); els[id] = e; }
    return els[id];
  },
  createElement(tag){ return { tag, href: '', download: '', click(){} }; },
  body: { appendChild(){}, removeChild(){} }
};
global.Blob = class { constructor(parts){ this.parts = parts; } };
global.URL = { createObjectURL(){ return 'blob:x'; }, revokeObjectURL(){} };

/* 4. 执行页面脚本并暴露内部函数 */
script = script.replace(/\}\)\(\);\s*$/, 'globalThis.__t = { compute, fmt, fmtGB, applyPreset, presetToCustom, normalizeCfg, importFromText, buildExportConfig, exportConfig, applyValues, computeComm, fmtMB };\n})();');
eval(script);

els['preset'].value = 'custom';
els['preset'].options = [{textContent:'✏️ 自定义配置'}, {textContent:'★ DeepSeekV4 10T(预估)'}];
els['preset'].selectedIndex = 0;

const readNum = id => els[id].textContent;
let fail = 0;
function assertEq(name, actual, expected) {
  if (actual !== expected) { console.error('[FAIL]', name, '=>', JSON.stringify(actual), 'expected', JSON.stringify(expected)); fail = 1; }
  else console.log('[OK]', name, '=', actual);
}
function assertContains(name, actual, needle) {
  if (!String(actual).includes(needle)) { console.error('[FAIL]', name, 'does not contain', needle, '=>', actual); fail = 1; }
  else console.log('[OK]', name, 'contains', JSON.stringify(needle));
}
function assertCls(name, el, cls) {
  if (!els[el].classList.contains(cls)) { console.error('[FAIL]', name, 'missing class', cls); fail = 1; }
  else console.log('[OK]', name, 'has class', cls);
}

/* Pass A: 默认 = DeepSeekV4 10T 预设（含 FP32 换算、×层数、Layer Input） */
__t.compute();
assertEq('kpi-cards',  readNum('kpi-cards'),  '2,048 卡');
assertContains('kpi-cards-sub 公式', readNum('kpi-cards-sub'), 'FSDP×CP = 2,048');
assertContains('kpi-cards-sub 说明', readNum('kpi-cards-sub'), '取最大');
assertEq('kpi-totalw', readNum('kpi-totalw'), '30,784.34 GB（30.06 TB）');
assertEq('kpi-cardw',  readNum('kpi-cardw'),  '15 GB');
assertEq('kpi-total',   readNum('kpi-total'),  '74.76 GB');
assertEq('cluster-static', readNum('cluster-static'), '123,137.37 GB（120.25 TB）');
assertEq('cluster-total',  readNum('cluster-total'),  '123,455.18 GB（120.56 TB）');
assertEq('card-total',     readNum('card-total'),     '74.76 GB');
assertEq('cluster-list rows', (els['cluster-list'].innerHTML.match(/<li>/g)||[]).length + '', '4');
assertEq('card-list rows',    (els['card-list'].innerHTML.match(/<li>/g)||[]).length + '', '7');
assertEq('kpi-hbm', readNum('kpi-hbm'), '占 96 GB HBM：77.9%');
assertContains('kpi-totalw-sub', readNum('kpi-totalw-sub'), '8,263.6 B');
assertContains('card-list Layer Input', els['card-list'].innerHTML, 'Layer Input');
assertContains('card-list Layer Input 值', els['card-list'].innerHTML, '1.67 GB');
console.log('--- Pass A OK ---\n');

/* Pass B: recompute=false */
els['recompute'].checked = false;
__t.compute();
assertEq('kpi-total (no recompute)', readNum('kpi-total'), '387.36 GB');
assertEq('card-total (no recompute)', readNum('card-total'), '387.36 GB');
assertEq('kpi-hbm (no recompute)', readNum('kpi-hbm'), '占 96 GB HBM：403.5%');
console.log('--- Pass B OK ---\n');

/* Pass C: HBM 超限 */
els['hbm'].value = '20';
__t.compute();
assertEq('kpi-hbm (hbm=20)', readNum('kpi-hbm'), '占 20 GB HBM：1,936.8%');
console.log('--- Pass C OK ---\n');

/* Pass D: 非法并行度 */
els['fsdp'].value = '0';
els['hbm'].value = '96';
__t.compute();
assertCls('warn shown', 'warn', 'show');
assertCls('fsdp invalid', 'fsdp', 'invalid');
assertContains('warn text', els['warn'].textContent, 'FSDP 必须 ≥ 1');
console.log('--- Pass D OK ---\n');

/* Pass E: 极小模型边界（seqLen/hidden 保留默认 2048/7168） */
for (const kv of [['attn','2'],['exp','0'],['nExp','1'],['layers','4'],['topk','1'],['act','1'],['fsdp','1'],['ep','1'],['efsdp','1'],['cp','1'],['hbm','100']]) els[kv[0]].value = kv[1];
els['recompute'].checked = true; els['swap'].checked = false;
__t.compute();
assertEq('minimal cards', readNum('kpi-cards'), '1 卡');
assertEq('minimal card-total', readNum('card-total'), '135.22 GB');
console.log('--- Pass E OK ---\n');

/* Pass F: 配置管理 */
els['attn'].value = '1'; els['fsdp'].value = '1'; els['recompute'].checked = false;
__t.applyPreset('deepseek-v4-10t');
assertEq('preset attn', els['attn'].value, 0.301);
assertEq('preset fsdp', els['fsdp'].value, 2048);
assertEq('preset seqLen', els['seqLen'].value, 2048);
assertEq('preset hidden', els['hidden'].value, 7168);
assertEq('preset moeHidden', els['moeHidden'].value, 3072);
assertEq('preset intraBW', els['intraBW'].value, 100);
assertEq('preset recompute', els['recompute'].checked + '', 'true');
assertEq('preset cards', readNum('kpi-cards'), '2,048 卡');
assertContains('preset msg', els['cfg-msg'].textContent, 'DeepSeekV4 10T(预估)');

els['preset'].value = 'deepseek-v4-10t';
__t.presetToCustom();
assertEq('presetToCustom 生效', els['preset'].value, 'custom');

const nested = __t.normalizeCfg({ model:{attnB:0.5,expertB:0.1,numExperts:64,layers:8,seqLen:4096,hiddenSize:4096,topk:2,activationGB:1.5},
                                  parallel:{fsdp:8,ep:4,efsdp:2,cp:2,hbmGB:96},
                                  options:{recompute:false,swapOptimizer:true} });
assertEq('nested cfg attn', nested.attn, 0.5);
assertEq('nested cfg seqLen', nested.seqLen, 4096);
assertEq('nested cfg hidden', nested.hidden, 4096);
assertEq('nested cfg swap', nested.swap, true);
assertEq('nested cfg recompute', nested.recompute, false);
const flat = __t.normalizeCfg({ attn:1, exp:0.2, nExp:10, layers:4, topk:1, act:2, seqLen:1024, hidden:512, fsdp:2, ep:2, efsdp:1, cp:1, hbm:80, recompute:true });
assertEq('flat cfg attn', flat.attn, 1);
assertEq('flat cfg hbm', flat.hbm, 80);
assertEq('flat cfg fsdpDtype 缺省', flat.fsdpDtype, 2);
assertEq('bad cfg (缺字段)', __t.normalizeCfg({foo:'bar'}), null);
assertEq('bad cfg (非法数值)', __t.normalizeCfg({attn:1,exp:1,nExp:0,layers:1,fsdp:1,ep:1,efsdp:1,cp:1}), null);

els['attn'].value = '9';
__t.importFromText(JSON.stringify({ name:'我的配置', model:{attnB:0.5,expertB:0.1,numExperts:64,layers:8,seqLen:2048,hiddenSize:7168,topk:2,activationGB:1.5},
                                    parallel:{fsdp:8,ep:4,efsdp:2,cp:2,hbmGB:96}, options:{recompute:true,swapOptimizer:false} }));
assertEq('import attn', els['attn'].value, 0.5);
assertEq('import seqLen', els['seqLen'].value, 2048);
assertEq('import select→custom', els['preset'].value, 'custom');
assertContains('import msg', els['cfg-msg'].textContent, '我的配置');
assertContains('import msg ok', els['cfg-msg'].className, 'ok');
assertEq('import 非法JSON 返回', __t.importFromText('{not json'), false);
assertContains('import msg err', els['cfg-msg'].className, 'err');

els['preset'].value = 'deepseek-v4-10t';
els['preset'].selectedIndex = 1;
__t.applyValues({ attn:0.301, exp:0.066, nExp:2048, layers:61, seqLen:2048, hidden:7168, moeHidden:3072, topk:30, act:5.21, fsdp:2048, ep:128, efsdp:16, cp:1, hbm:96, intraBW:100, intraSize:16, superBW:200, superSize:1024, interBW:50, fsdpDtype:2, recompute:true, swap:false });
const cfgObj = __t.buildExportConfig();
const expected = { name:'DeepSeekV4 10T(预估)', version:1,
  model:{attnB:0.301,expertB:0.066,numExperts:2048,layers:61,seqLen:2048,hiddenSize:7168,moeHidden:3072,topk:30,activationGB:5.21},
  parallel:{fsdp:2048,ep:128,efsdp:16,cp:1,hbmGB:96},
  network:{intraBW:100,intraSize:16,superBW:200,superSize:1024,interBW:50,fsdpDtype:2},
  options:{recompute:true,swapOptimizer:false} };
assertEq('export cfg JSON', JSON.stringify(cfgObj), JSON.stringify(expected));
els['preset'].value = 'custom';
__t.applyValues({ attn:0.301, exp:0.066, nExp:2048, layers:61, seqLen:2048, hidden:7168, moeHidden:3072, topk:30, act:5.21, fsdp:2048, ep:128, efsdp:16, cp:1, hbm:96, intraBW:100, intraSize:16, superBW:200, superSize:1024, interBW:50, fsdpDtype:2, recompute:true, swap:false });
assertEq('export 自定义 name', __t.buildExportConfig().name, '自定义配置');
__t.exportConfig();
assertContains('export msg', els['cfg-msg'].textContent, '已导出配置');
console.log('--- Pass F OK ---\n');

/* Pass G: SwapOptimizer=True → 仅单卡优化器=0，集群口径不变 */
els['swap'].checked = true;
__t.compute();
assertEq('swap 单卡总显存', readNum('kpi-total'), '44.76 GB');
assertEq('swap card-total', readNum('card-total'), '44.76 GB');
assertEq('swap kpi-hbm', readNum('kpi-hbm'), '占 96 GB HBM：46.6%');
assertEq('swap cluster-static 不变', readNum('cluster-static'), '123,137.37 GB（120.25 TB）');
assertEq('swap cluster-total 不变',  readNum('cluster-total'),  '123,455.18 GB（120.56 TB）');
assertContains('swap 单卡优化器=0', els['card-list'].innerHTML, 'class="val">0 GB');
assertContains('swap 单卡标注', els['card-list'].innerHTML, 'SwapOptimizer：交换出卡');
assertContains('swap 集群优化器保留', els['cluster-list'].innerHTML, '61,568.68 GB');
if (els['cluster-list'].innerHTML.includes('SwapOptimizer')) { console.error('[FAIL] 集群不应出现 Swap 标注'); fail = 1; }
else console.log('[OK] 集群无 Swap 标注');
els['swap'].checked = false;
__t.compute();
assertEq('swap 关闭后恢复', readNum('card-total'), '74.76 GB');
console.log('--- Pass G OK ---\n');

/* Pass H: 通信耗时 EP Alltoall 列（DeepSeekV4 预设，机内16/超节点1024/集群10240） */
__t.compute();
assertEq('EP 列 总接受数据', readNum('comm-a-total'), '0.3 GB');
assertEq('EP 列 每份数据', readNum('comm-a-piece'), '3 MB');
assertEq('EP 列 有效带宽', readNum('comm-a-bw'), '178.87 GB/s');
assertEq('EP 列 耗时', readNum('comm-a-time'), '1.950 ms');
assertEq('EP 列 瓶颈层', readNum('comm-a-bottleneck'), '超节点内（112 份 · 200 GB/s）');
assertEq('EP 列 tiers 行数', (els['comm-a-tiers'].innerHTML.match(/<li>/g)||[]).length + '', '3');
assertContains('EP 列 机内', els['comm-a-tiers'].innerHTML, '15 份');
assertContains('EP 列 机内占比', els['comm-a-tiers'].innerHTML, '11.8%');
assertContains('EP 列 超节点内', els['comm-a-tiers'].innerHTML, '112 份');
assertContains('EP 列 超节点内占比', els['comm-a-tiers'].innerHTML, '88.2%');
assertContains('EP 列 超节点间', els['comm-a-tiers'].innerHTML, '0 份');
if (els['comm-a-warn'].classList.contains('show')) { console.error('[FAIL] comm-a-warn 不应出现'); fail = 1; }
else console.log('[OK] EP 列 comm-warn 缺省隐藏');
console.log('--- Pass H OK ---\n');

/* Pass I: EP=2048 跨超节点分解 */
els['ep'].value = '2048';
__t.compute();
assertContains('ep2048 机内', els['comm-a-tiers'].innerHTML, '15 份');
assertContains('ep2048 超节点内', els['comm-a-tiers'].innerHTML, '1,008 份');
assertContains('ep2048 超节点间', els['comm-a-tiers'].innerHTML, '1,024 份');
assertEq('ep2048 瓶颈层', readNum('comm-a-bottleneck'), '超节点间（1,024 份 · 50 GB/s）');
els['ep'].value = '128';
__t.compute();
console.log('--- Pass I OK ---\n');

/* Pass J: 通信对比卡片两列并排（EP Alltoall 与 FSDP Allgather/RS），EFSDP 列预留 */
els['fsdp'].value = '2048'; els['fsdpDtype'].value = '2';  /* 默认 BF16 */
els['intraSize'].value = '16'; els['superSize'].value = '1024';
els['recompute'].checked = true; els['swap'].checked = false;
__t.compute();
/* 左列 EP Alltoall */
assertEq('EP 列 总接受数据', readNum('comm-a-total'), '0.3 GB');
assertEq('EP 列 每份数据', readNum('comm-a-piece'), '3 MB');
assertEq('EP 列 算法带宽', readNum('comm-a-bw'), '178.87 GB/s');
assertEq('EP 列 耗时', readNum('comm-a-time'), '1.950 ms');
assertEq('EP 列 瓶颈层', readNum('comm-a-bottleneck'), '超节点内（112 份 · 200 GB/s）');
/* 右列 FSDP Allgather/RS（默认 dtype=2 即 BF16：0.301B × 10⁹ × 2 / 1024² / 2048 × 2047 ≈ 573.8 MB） */
assertEq('FSDP 通信域', __t.computeComm('fsdp').domain + '', '2048');
assertEq('FSDP 列 总接收(BF16)', readNum('comm-f-total'), '0.6 GB');
assertEq('FSDP 列 每份(BF16)', readNum('comm-f-piece'), '287 KB');
assertEq('FSDP 列 算法带宽', readNum('comm-f-bw'), '103.23 GB/s');
assertEq('FSDP 列 耗时(BF16)', readNum('comm-f-time'), '5.429 ms');
assertEq('FSDP 列 瓶颈层', readNum('comm-f-bottleneck'), '机内（1,920 份 · 100 GB/s）');
assertContains('FSDP 列 机内', els['comm-f-tiers'].innerHTML, '1,920 份');
assertContains('FSDP 列 机内占比', els['comm-f-tiers'].innerHTML, '93.8%');
assertContains('FSDP 列 超节点内', els['comm-f-tiers'].innerHTML, '126 份');
assertContains('FSDP 列 超节点间', els['comm-f-tiers'].innerHTML, '1 份');
assertContains('EP 列 机内', els['comm-a-tiers'].innerHTML, '15 份');
assertContains('EP 列 超节点内', els['comm-a-tiers'].innerHTML, '112 份');
if (els['comm-a-warn'].classList.contains('show')) { console.error('[FAIL] EP 列 warn 不应出现'); fail = 1; }
else console.log('[OK] EP 列 comm-warn 缺省隐藏');
if (els['comm-f-warn'].classList.contains('show')) { console.error('[FAIL] FSDP 列 warn 不应出现'); fail = 1; }
else console.log('[OK] FSDP 列 comm-warn 缺省隐藏');
/* EFSDP 第三列（通信域=EFSDP=16；有效层级 = 物理大小/EP → 超节点 1024/128=8、机内 16/128→1；
   载荷 = 0.066×2048/128 × 10⁹×2 / 1024² × 15/16 ≈ 1,888.3 MB = 1.8 GB，默认 BF16） */
assertEq('EFSDP 通信域', __t.computeComm('efsdp').domain + '', '16');
assertEq('EFSDP 列 总接受数据', readNum('comm-e-total'), '1.8 GB');
assertEq('EFSDP 列 每份数据', readNum('comm-e-piece'), '126 MB');
assertEq('EFSDP 列 算法带宽', readNum('comm-e-bw'), '214.29 GB/s');
assertEq('EFSDP 列 耗时', readNum('comm-e-time'), '8.605 ms');
assertEq('EFSDP 列 瓶颈层', readNum('comm-e-bottleneck'), '超节点内（14 份 · 200 GB/s）');
assertContains('EFSDP 列 机内', els['comm-e-tiers'].innerHTML, '0 份');
assertContains('EFSDP 列 超节点内', els['comm-e-tiers'].innerHTML, '14 份');
assertContains('EFSDP 列 超节点间', els['comm-e-tiers'].innerHTML, '1 份');
assertContains('EFSDP 列 公式文案', els['comm-e-total-sub'].textContent, '/EP');
if (els['comm-e-warn'].classList.contains('show')) { console.error('[FAIL] EFSDP 列 warn 不应出现'); fail = 1; }
else console.log('[OK] EFSDP 列 comm-warn 缺省隐藏');
/* 小通信域 FSDP=256：机内 = 255−15−0 = 240 份 */
els['fsdp'].value = '256';
__t.compute();
assertContains('fsdp256 机内', els['comm-f-tiers'].innerHTML, '240 份');
assertContains('fsdp256 超节点内', els['comm-f-tiers'].innerHTML, '15 份');
assertContains('fsdp256 超节点间', els['comm-f-tiers'].innerHTML, '0 份');
els['fsdp'].value = '2048';
__t.compute();

/* dtype 选项切换：FP32=4 → 载荷翻倍；FP8=1 → 载荷减半；EP 列不受影响 */
els['fsdpDtype'].value = '4';
__t.compute();
assertEq('FSDP FP32 总接收', readNum('comm-f-total'), '1.1 GB');
assertEq('FSDP FP32 每份', readNum('comm-f-piece'), '574 KB');
assertEq('FSDP FP32 耗时', readNum('comm-f-time'), '10.857 ms');
assertEq('EP 列不受 dtype 影响', readNum('comm-a-total'), '0.3 GB');
els['fsdpDtype'].value = '1';
__t.compute();
assertEq('FSDP FP8 总接收', readNum('comm-f-total'), '0.3 GB');
assertEq('FSDP FP8 每份', readNum('comm-f-piece'), '144 KB');
assertEq('FSDP FP8 耗时', readNum('comm-f-time'), '2.714 ms');
els['fsdpDtype'].value = '2';
__t.compute();
assertEq('FSDP 恢复 BF16 默认', readNum('comm-f-total'), '0.6 GB');
console.log('--- Pass J OK ---\n');

/* Pass K: 外部预设文件 presets.json 与页面内置 fallback 一致性 */
const pcfg = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'presets.json'), 'utf8'));
assertEq('presets.json default.moeHidden', pcfg.default.moeHidden, 3072);
assertEq('presets.json default.intraBW', pcfg.default.intraBW, 100);
assertEq('presets.json default.hidden', pcfg.default.hidden, 7168);
assertEq('presets.json 预设数', (pcfg.presets || []).length + '', '1');
assertEq('presets.json 预设 id', pcfg.presets[0].id, 'deepseek-v4-10t');
assertEq('presets.json 预设 moeHidden', pcfg.presets[0].values.moeHidden, 3072);
assertEq('presets.json 预设 intraBW', pcfg.presets[0].values.intraBW, 100);
__t.applyPreset('deepseek-v4-10t');
assertEq('外部预设 moeHidden 生效', els['moeHidden'].value, 3072);
assertEq('外部预设 intraBW 生效', els['intraBW'].value, 100);
assertEq('外部预设 hidden 生效', els['hidden'].value, 7168);
console.log('--- Pass K OK ---\n');

if (fail) { console.error('TESTS FAILED'); process.exit(1); }
console.log('ALL TESTS PASSED');