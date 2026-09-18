$ErrorActionPreference = 'Stop'
$rootDir = $PSScriptRoot
$bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (Test-Path -LiteralPath $bundledPython) {
    $pythonPath = $bundledPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) { throw '需要 Python 3.10+；请安装后重试。' }
    $pythonPath = $pythonCommand.Source
}
& $pythonPath -X utf8 (Join-Path $rootDir '03_代码实验/test_core.py')
if ($LASTEXITCODE -ne 0) { throw '教学代码校验失败，未继续生成。' }
& $pythonPath -X utf8 (Join-Path $rootDir '03_代码实验/run_examples.py')
if ($LASTEXITCODE -ne 0) { throw '教学示例运行失败。' }
& $pythonPath -X utf8 (Join-Path $rootDir '03_代码实验/test_distillation.py')
if ($LASTEXITCODE -ne 0) { throw '蒸馏数值测试失败。' }
& $pythonPath -X utf8 (Join-Path $rootDir '03_代码实验/distillation_examples.py')
if ($LASTEXITCODE -ne 0) { throw '蒸馏教学示例失败。' }
& $pythonPath -X utf8 (Join-Path $rootDir '90_维护与来源/test_rendering.py')
if ($LASTEXITCODE -ne 0) { throw '公式转换回归检查失败。' }
& $pythonPath -X utf8 (Join-Path $rootDir '03_代码实验/test_posttraining_pretraining.py')
if ($LASTEXITCODE -ne 0) { throw '后训练与预训练数值检查失败。' }
& $pythonPath -X utf8 (Join-Path $rootDir '03_代码实验/posttraining_pretraining_examples.py')
if ($LASTEXITCODE -ne 0) { throw '后训练与预训练教学示例失败。' }
& $pythonPath -X utf8 (Join-Path $rootDir '03_代码实验/test_agent_reliability.py')
if ($LASTEXITCODE -ne 0) { throw 'Agent 可靠性模拟检查失败。' }
& $pythonPath -X utf8 (Join-Path $rootDir '03_代码实验/agent_reliability_examples.py')
if ($LASTEXITCODE -ne 0) { throw 'Agent 教学模拟失败。' }
$transformerPython = $null
foreach ($candidate in @((Join-Path $rootDir '.venv-transformer/Scripts/python.exe'), (Join-Path $rootDir 'work/transformer-visual-revision/venv/Scripts/python.exe'), $pythonPath)) {
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    & $candidate -c 'import importlib.util; raise SystemExit(0 if importlib.util.find_spec("torch") else 1)'
    if ($LASTEXITCODE -eq 0) { $transformerPython = $candidate; break }
}
if (-not $transformerPython) { throw 'Transformer 检查需要 PyTorch；请按“03_代码实验/Transformer手写运行与练习.md”安装独立环境后重试。阅读 HTML 无需安装。' }
& $transformerPython -X utf8 (Join-Path $rootDir '03_代码实验/test_transformer_handwrite.py')
if ($LASTEXITCODE -ne 0) { throw 'Transformer 数值、梯度或行为检查失败。' }
& $transformerPython -X utf8 (Join-Path $rootDir '03_代码实验/transformer_handwrite.py') | Set-Content -LiteralPath (Join-Path $rootDir '03_代码实验/Transformer手写运行结果.json') -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw 'Transformer 玩具训练失败。' }
& $pythonPath -X utf8 (Join-Path $rootDir '90_维护与来源/build.py')
if ($LASTEXITCODE -ne 0) { throw '知识库结构或数学转换校验失败。' }
Write-Output '验证与重建完成。双击“开始阅读.html”即可离线阅读。'
