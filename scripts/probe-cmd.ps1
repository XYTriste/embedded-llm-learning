[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$procs = Get-CimInstance Win32_Process
$ids = 28756, 43320, 68092, 54252
$lines = foreach ($id in $ids) {
  $p = $procs | Where-Object { $_.ProcessId -eq $id }
  if (-not $p) { "PID $id : exited"; continue }
  $par = $procs | Where-Object { $_.ProcessId -eq $p.ParentProcessId }
  "PID $id  parent: $($par.Name)($($par.ProcessId))  started: $($p.CreationDate)  cmdline: $($p.CommandLine)"
  $kids = $procs | Where-Object { $_.ParentProcessId -eq $id }
  if ($kids) {
    foreach ($k in $kids) { "    child: $($k.Name)($($k.ProcessId))  $($k.CommandLine)" }
  } else {
    "    (no child processes)"
  }
}
$lines | Out-File -Encoding utf8 "probe-cmd-out.txt"
