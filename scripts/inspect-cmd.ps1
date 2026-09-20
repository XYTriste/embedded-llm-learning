# 列出本机所有 cmd / powershell / 终端进程及其子进程
$procs = Get-CimInstance Win32_Process
$map = @{}
foreach ($p in $procs) { $map[$p.ProcessId] = $p }

$shells = $procs | Where-Object { $_.Name -match '^(cmd|powershell|pwsh|WindowsTerminal|conhost|mintty)\.exe$' }

$result = foreach ($s in $shells) {
  $title = ''
  try { $title = (Get-Process -Id $s.ProcessId -ErrorAction Stop).MainWindowTitle } catch { }
  $parentName = ''
  if ($map.ContainsKey($s.ParentProcessId)) { $parentName = $map[$s.ParentProcessId].Name }
  $kids = $procs | Where-Object { $_.ParentProcessId -eq $s.ProcessId }
  $kidDesc = ($kids | ForEach-Object {
    $cl = ($_.CommandLine -replace '\s+', ' ')
    if ($cl.Length -gt 160) { $cl = $cl.Substring(0, 160) + '...' }
    "    {0}({1}): {2}" -f $_.Name, $_.ProcessId, $cl
  }) -join "`n"
  [PSCustomObject]@{
    Name     = $s.Name
    PID      = $s.ProcessId
    Parent   = "$parentName($($s.ParentProcessId))"
    Started  = $s.CreationDate
    Title    = $title
    CmdLine  = $s.CommandLine
    Children = $kidDesc
  }
}
$result | Format-List
