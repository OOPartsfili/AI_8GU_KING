/* Offline iOS reader web regression. Uses a fresh browser context, never a user profile.
 * Run after python ios/tools/prepare_assets.py: node ios/tests/check_reader.cjs
 * BROWSER_ENGINE=webkit can select an installed Playwright WebKit for additional coverage.
 * This browser test does not replace signing/installing and testing the native app on iPhone.
 */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
let playwright;
try { playwright = require('playwright'); }
catch { playwright = require(path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright')); }
const root = path.resolve(__dirname, '../..');
const reader = path.join(root, 'ios/AI8GUKing/Resources/reader.html');
const output = process.env.IOS_QA_OUTPUT || path.join(root, 'work/ios/qa');
const engine = process.env.BROWSER_ENGINE || 'chromium';
const url = pathToFileURL(reader).href;
const widths = [{width:393, height:852}, {width:852, height:393}];
fs.mkdirSync(output, {recursive:true});
assert(fs.existsSync(reader), 'Generate ios/AI8GUKing/Resources/reader.html before running web QA.');

async function checkOverflow(page, name) {
  const overflow = await page.evaluate(() => ({width:innerWidth, scroll:document.documentElement.scrollWidth}));
  assert(overflow.scroll <= overflow.width + 1, `${name}: horizontal page overflow ${JSON.stringify(overflow)}`);
}
async function openOptions(page) {
  const details = page.locator('#mobileOptions');
  if (!await details.evaluate(el => el.open)) await details.locator(':scope > summary').click();
}
async function closeOptions(page) {
  const details = page.locator('#mobileOptions');
  if (await details.evaluate(el => el.open)) await details.locator(':scope > summary').click();
}
async function go(page, hash, selector) {
  await page.evaluate(hash => go(hash), hash);
  await page.waitForFunction(hash => location.hash === '#' + hash, hash);
  if (selector) await page.locator(selector).first().waitFor();
}

(async () => {
  const launch = {headless:true};
  if (process.env.KNOWLEDGE_BROWSER) launch.executablePath = process.env.KNOWLEDGE_BROWSER;
  else if (engine === 'chromium' && process.platform === 'win32') launch.executablePath = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
  const browser = await playwright[engine].launch(launch);
  const report = {
    method:`Playwright ${engine}; isolated context; local file; offline networking; native bridge mocked. Physical iPhone and WKWebView signing are separate checks.`,
    viewports:widths, questions:0, question_route_checks:0, math_nodes:0, visualization_checks:0,
    errors:[], automatic_http_requests:[], screenshots:[],
  };
  let page;
  try {
    const context = await browser.newContext({viewport:widths[0], isMobile:true, hasTouch:true, offline:true});
    await context.addInitScript(() => {
      window.__nativeMessages = [];
      window.webkit = {messageHandlers:{nativeBridge:{postMessage(message) {
        window.__nativeMessages.push(JSON.parse(JSON.stringify(message)));
      }}}};
    });
    page = await context.newPage();
    page.on('pageerror', error => report.errors.push(error.message));
    page.on('request', request => { if (/^https?:/i.test(request.url())) report.automatic_http_requests.push(request.url()); });
    await page.goto(url);
    await page.locator('.hero').waitFor();
    const data = await page.evaluate(() => ({
      count:DB.stats.question_count, ids:DB.questions.map(q => q.id),
      beginner:DB.beginner, core:DB.core, sources:DB.sources.length,
    }));
    assert.equal(data.count, 341, 'All current question records must ship offline.');
    assert.equal(data.ids.length, data.count);
    assert.equal(new Set(data.ids).size, data.count);
    report.questions = data.count;
    report.sources = data.sources;
    assert.equal(await page.locator('.qbody').count(), 0, 'Home should not eagerly render question bodies.');
    assert.equal(await page.locator('script[id^="data-q-"]').count(), data.count, 'Home should retain lazy records.');
    assert.equal(await page.locator('#mobileNavToggle').getAttribute('aria-expanded'), 'false');
    assert.equal(await page.locator('#nav').isVisible(), false, 'Chapter navigation starts collapsed.');
    assert.equal(await page.locator('#mobileOptions').evaluate(el => el.open), false, 'Extra controls start collapsed.');
    const aboveFold = await page.evaluate(() => ({
      searchBottom:document.getElementById('search').getBoundingClientRect().bottom,
      contentTop:document.getElementById('content').getBoundingClientRect().top,
      height:innerHeight,
      searchFont:parseFloat(getComputedStyle(document.getElementById('search')).fontSize),
    }));
    assert(aboveFold.searchBottom < aboveFold.height / 2, 'Search must be visible in the first half of the phone screen.');
    assert(aboveFold.contentTop < aboveFold.height * 0.65, 'Menus must leave space for actual reading above the fold.');
    assert(aboveFold.searchFont >= 16, 'Search uses 16 px or larger type to avoid iOS focus zoom.');
    await checkOverflow(page, 'home portrait');
    await page.screenshot({path:path.join(output, 'home-393.png')});
    report.screenshots.push('home-393.png');
    report.compact_home = aboveFold;

    // Exercise real navigation controls, rather than calling route() for this contract.
    await page.locator('#mobileNavToggle').click();
    assert.equal(await page.locator('#mobileNavToggle').getAttribute('aria-expanded'), 'true');
    await page.locator('#nav button[data-route="beginner"]').click();
    await page.locator('#q-' + data.beginner[0]).waitFor();
    assert.deepEqual(await page.locator('.question').evaluateAll(nodes => nodes.map(n => n.id.slice(2))), data.beginner);
    assert.equal(await page.locator('#mobileNavToggle').getAttribute('aria-expanded'), 'false', 'Selecting a chapter closes navigation.');
    assert.equal(await page.locator('.qbody').count(), 0, 'Collapsed courses keep body rendering lazy.');
    await page.locator('#mobileQuickNav button').first().waitFor();
    for (const [id, count, first] of [['transformerBtn',22,'TH01'], ['agentBtn',50,'A01'], ['distillBtn',34,'KD01'], ['coreBtn',data.core.length,data.core[0]]]) {
      await openOptions(page);
      await page.locator('#' + id).click();
      await page.locator('#q-' + first).waitFor();
      assert.equal(await page.locator('.question').count(), count, id);
      await closeOptions(page);
      await checkOverflow(page, id);
      if (id === 'transformerBtn') {
        await page.locator('[data-code="transformer_handwrite.py"]').click();
        const codeMessage = await page.evaluate(() => window.__nativeMessages.find(message => message.action === 'export' && message.filename === 'transformer_handwrite.py'));
        assert(codeMessage, 'The tutorial code download must use the native share bridge.');
        assert.equal(codeMessage.mimeType, 'text/plain');
        assert.equal(codeMessage.content, fs.readFileSync(path.join(root, '03_代码实验/transformer_handwrite.py'), 'utf8').replace(/\r\n/g, '\n'));
        report.code_export_matches_source = true;
      }
    }
    report.courses = {beginner:20, transformer:22, agent:50, distillation:34, core:data.core.length};

    // Search must show usable results without reopening the large navigation/filter blocks.
    await page.locator('#search').fill('KV cache');
    await page.locator('#searchBtn').click();
    await page.getByRole('heading', {name:'检索结果', exact:true}).waitFor();
    assert(await page.locator('.question').count() >= 3);
    await page.evaluate(() => scrollTo(0, 0));
    const resultTop = await page.locator('.question').first().evaluate(el => el.getBoundingClientRect().top);
    assert(resultTop < 852, 'First search result must be visible on the phone without a screen of menus.');
    await checkOverflow(page, 'search portrait');
    await page.screenshot({path:path.join(output, 'search-393.png')});
    report.screenshots.push('search-393.png');
    report.search = {count:await page.locator('.question').count(), first_result_top:resultTop};
    await page.locator('#clearBtn').click();

    // Deep reading, persisted progress, and the actual buttons used by WKWebView.
    await go(page, 'q-M02', '#q-M02 .qbody');
    await page.locator('#q-M02 .deep > summary').click();
    await page.locator('#q-M02 math').first().waitFor();
    assert((await page.locator('#q-M02 .deepbody').innerText()).length > 100);
    await page.locator('.study').selectOption('done');
    await page.reload();
    await page.locator('.study').waitFor();
    assert.equal(await page.locator('.study').inputValue(), 'done');
    await openOptions(page);
    await page.locator('#exportBtn').click();
    const exportedMessage = await page.evaluate(() => window.__nativeMessages.find(message => message.action === 'export' && message.mimeType === 'application/json'));
    assert(exportedMessage, 'Export button must use the native share bridge.');
    assert.equal(exportedMessage.mimeType, 'application/json');
    assert(exportedMessage.filename.endsWith('.json'));
    const exported = JSON.parse(exportedMessage.content);
    assert.equal(exported.schema, 'llm-study-progress-v1');
    assert.equal(exported.progress.M02.state, 'done');
    await page.locator('#importBtn').click();
    assert(await page.evaluate(() => window.__nativeMessages.some(message => message.action === 'importProgress')), 'Import button must open the native document picker.');
    await closeOptions(page);
    await page.locator('.study').selectOption('new');
    const imported = await page.evaluate(raw => window.AI8GU.importProgress(raw), exportedMessage.content);
    assert.equal(imported.ok, true);
    assert.equal(await page.locator('.study').inputValue(), 'done');
    await page.reload();
    await page.locator('.study').waitFor();
    assert.equal(await page.locator('.study').inputValue(), 'done', 'Imported progress must survive a reload.');
    for (const invalid of ['{broken', '{}', '{"schema":"wrong","progress":{"M02":{"state":"new"}}}']) {
      const result = await page.evaluate(raw => window.AI8GU.importProgress(raw), invalid);
      assert.equal(result.ok, false, 'Malformed progress should be reported, not throw or overwrite data.');
      assert.equal(await page.locator('.study').inputValue(), 'done');
    }
    const scrubbed = await page.evaluate(() => window.AI8GU.importProgress(JSON.stringify({
      schema:'llm-study-progress-v1', progress:{M02:{state:'invalid'}, UNKNOWN_QUESTION:{state:'done'}, TH01:{state:'review'}},
    })));
    assert.equal(scrubbed.ok, true);
    const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('llm-interview-study-v1')));
    assert.equal(saved.M02.state, 'done');
    assert.equal(saved.TH01.state, 'review');
    assert.equal(Object.hasOwn(saved, 'UNKNOWN_QUESTION'), false);
    report.progress_reload_export_import = true;
    report.invalid_import_preserves_progress = true;

    // Every question, including deep MathML, must remain readable at both phone orientations.
    await openOptions(page);
    await page.locator('#depth').selectOption('full');
    await closeOptions(page);
    const expectedMath = await page.evaluate(() => Object.fromEntries(DB.questions.map(question => {
      const parsed = document.createElement('div');
      parsed.innerHTML = question.html;
      return [question.id, parsed.querySelectorAll('math').length];
    })));
    const mathCounts = new Map();
    for (const viewport of widths) {
      await page.setViewportSize(viewport);
      for (const id of data.ids) {
        await go(page, 'q-' + id, '#q-' + id + ' .qbody');
        const state = await page.locator('#q-' + id).evaluate(el => ({
          length:el.innerText.length, math:el.querySelectorAll('math').length,
          mathErrors:el.querySelectorAll('merror').length,
          deep:el.querySelector('.deep')?.open,
          width:document.documentElement.scrollWidth,
          viewport:innerWidth,
        }));
        assert(state.length > 100, id + ' has readable offline content');
        assert.equal(state.mathErrors, 0, id + ' MathML');
        assert.equal(state.math, expectedMath[id], id + ' must render every formula present in its source');
        assert(state.deep === true, id + ' deep reading content');
        assert(state.width <= state.viewport + 1, id + ' overflow at ' + viewport.width);
        mathCounts.set(id, state.math);
        report.question_route_checks++;
      }
      await go(page, 'q-M02', '#q-M02 math');
      await page.evaluate(() => scrollTo(0, 0));
      const shot = 'question-' + viewport.width + '.png';
      await page.screenshot({path:path.join(output, shot)});
      report.screenshots.push(shot);
      console.log(`Checked ${data.count} deep question routes at ${viewport.width}×${viewport.height}`);
    }
    report.math_nodes = [...mathCounts.values()].reduce((sum, count) => sum + count, 0);
    assert(report.math_nodes > 0, 'Bundled offline formulas must render.');

    const types = ['projection','attention','heads','block','rope','cache','probability','lora'];
    for (const viewport of widths) {
      await page.setViewportSize(viewport);
      for (const type of types) {
        await go(page, 'viz-' + type, '[data-mounted="' + type + '"]');
        for (const control of await page.locator('.viz-controls input[type="range"]').all()) {
          await control.evaluate(el => { el.value = el.max; el.dispatchEvent(new Event('input', {bubbles:true})); });
        }
        if (await page.locator('[data-cell]').count()) await page.locator('[data-cell]').last().click();
        const visualText = await page.locator('.viz-host').innerText();
        assert(!/\bNaN\b|undefined|Infinity/.test(visualText), type + ' finite outputs');
        await checkOverflow(page, type + ' ' + viewport.width);
        report.visualization_checks++;
      }
    }
    await page.setViewportSize(widths[0]);
    await go(page, 'agent', '#agentWalk');
    for (const scenario of ['normal','timeout','unknown']) {
      await page.locator('#agentScenario').selectOption(scenario);
      for (let i=0; i<4; i++) await page.locator('#agentNext').click();
      assert((await page.locator('.agent-walk-state').innerText()).includes(scenario === 'unknown' ? '暂停，结果仍未知' : '已验证完成'));
      assert(await page.locator('#agentNext').isDisabled());
      await page.locator('#agentRestart').click();
    }
    report.agent_scenarios = 3;
    await page.evaluate(() => window.AI8GU.goHome());
    await page.locator('.hero').waitFor();
    assert.deepEqual(report.errors, []);
    assert.deepEqual(report.automatic_http_requests, []);
    await page.evaluate(() => localStorage.clear());
    report.passed = true;
    fs.writeFileSync(path.join(output, 'reader-qa.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } catch (error) {
    report.passed = false;
    report.failure = error.stack || String(error);
    fs.writeFileSync(path.join(output, 'reader-qa.json'), JSON.stringify(report, null, 2));
    if (page) await page.screenshot({path:path.join(output, 'failure.png')}).catch(() => {});
    throw error;
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
