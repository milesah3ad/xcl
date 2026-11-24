param([string]$Work = "$env:TEMP\nv680", [string]$Tag = "dxc19", [string]$Dxc = $env:XCL_DXC_DIR, [string]$Rounds = "250,500,1000,1800", [string]$Shapes = "base,atomic,atomic64,loop,wave", [string]$Heap = "1,0", [string]$Seeds = "1..3")
# Build first: cargo build --release --example nv680. Run: powershell -ExecutionPolicy Bypass -File nv680_run.ps1 [-Shapes atomic64 -Heap 1,0 -Rounds 250,500]
# Results: <Work>\<Tag>.tsv (default Work: %TEMP%\nv680). Lists are comma-separated.
# Generate synthetic compute programs and build each pipeline in its own process,
# so a driver abort (0xC0000409 / exit 127) is recorded instead of ending the run.
New-Item -ItemType Directory -Force $Work | Out-Null; Set-Location $Work
$env:XCL_DXC_DIR = $Dxc
$x = Join-Path $PSScriptRoot "..\..\..\target\release\examples\nv680.exe"
$out = Join-Path $Work $Tag
New-Item -ItemType Directory -Force $out | Out-Null
$log = Join-Path $Work "$Tag.tsv"
"file`tdxil_kb`tresult`texit`tstderr" | Set-Content $log
foreach ($s in $Shapes.Split(",")) { foreach ($h in $Heap.Split(",")) { foreach ($r in $Rounds.Split(",")) {
    & $x gen --out $out --name "${s}_h${h}_r$r" --rounds $r --heap $h --shape $s --seeds $Seeds 2>&1 | Out-Null
} } }
Get-ChildItem "$out\*.dxil" | Sort-Object Length | ForEach-Object {
    $errf = "$out\$($_.BaseName).stderr"
    $p = Start-Process $x -ArgumentList "create", $_.FullName -NoNewWindow -Wait -PassThru -RedirectStandardOutput "$out\$($_.BaseName).stdout" -RedirectStandardError $errf
    $res = "$(Get-Content "$out\$($_.BaseName).stdout" -Raw)" -replace "`r?`n", " "
    $err = ("$(Get-Content $errf -Raw)" -replace "`r?`n", " ").Trim()
    $line = "{0}`t{1}`t{2}`t0x{3:X8}`t{4}" -f $_.Name, [int]($_.Length / 1KB), $res.Trim(), $p.ExitCode, $err
    Add-Content $log $line
    Write-Output $line
}
"DONE $Tag"
