[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$source = @"
using System;
using System.Runtime.InteropServices;
using System.Text;
public class ConReader2 {
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool FreeConsole();
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool AttachConsole(uint dwProcessId);
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] public static extern IntPtr CreateFile(string lpFileName, uint dwDesiredAccess, uint dwShareMode, IntPtr lpSecurityAttributes, uint dwCreationDisposition, uint dwFlagsAndAttributes, IntPtr hTemplateFile);
  [DllImport("kernel32.dll")] public static extern bool GetConsoleScreenBufferInfo(IntPtr hConsoleOutput, out CONSOLE_SCREEN_BUFFER_INFO lpConsoleScreenBufferInfo);
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode)] public static extern bool ReadConsoleOutputCharacter(IntPtr hConsoleOutput, [Out] StringBuilder lpCharacter, uint nLength, COORD dwReadCoord, out uint lpNumberOfCharsRead);

  public struct COORD { public short X, Y; public COORD(short x, short y){X=x;Y=y;} }
  public struct SMALL_RECT { public short Left, Top, Right, Bottom; }
  public struct CONSOLE_SCREEN_BUFFER_INFO {
    public COORD dwSize;
    public COORD dwCursorPosition;
    public ushort wAttributes;
    public SMALL_RECT srWindow;
    public COORD dwMaximumWindowSize;
  }

  public static string ReadScreen(uint pid) {
    FreeConsole();
    if (!AttachConsole(pid)) return "ATTACH_FAILED err=" + Marshal.GetLastWin32Error();
    try {
      IntPtr h = CreateFile("CONOUT$", 0x80000000 | 0x40000000, 0x00000001 | 0x00000002, IntPtr.Zero, 3, 0, IntPtr.Zero);
      if (h == new IntPtr(-1)) return "OPEN_CONOUT_FAILED err=" + Marshal.GetLastWin32Error();
      CONSOLE_SCREEN_BUFFER_INFO info;
      if (!GetConsoleScreenBufferInfo(h, out info)) return "BUFINFO_FAILED err=" + Marshal.GetLastWin32Error();
      short left = info.srWindow.Left, top = info.srWindow.Top;
      short width = (short)(info.srWindow.Right - info.srWindow.Left + 1);
      short height = (short)(info.srWindow.Bottom - info.srWindow.Top + 1);
      var sb = new StringBuilder();
      for (short y = top; y <= (short)(top + height); y++) {
        var line = new StringBuilder(width + 1);
        uint read;
        if (!ReadConsoleOutputCharacter(h, line, (uint)width, new COORD(left, y), out read)) { sb.AppendLine("<read failed>"); continue; }
        sb.AppendLine(line.ToString(0, (int)read).TrimEnd());
      }
      return sb.ToString();
    } finally { FreeConsole(); }
  }
}
"@
Add-Type -TypeDefinition $source

$self = $PID
$parent = (Get-CimInstance Win32_Process -Filter "ProcessId=$self").ParentProcessId
$cands = Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'powershell.exe' -or $_.Name -eq 'pwsh.exe') -and $_.ProcessId -ne $self -and $_.ProcessId -ne $parent }
$out = foreach ($c in $cands) {
  "===== $($c.Name) PID $($c.ProcessId)  started $($c.CreationDate) ====="
  "cmdline: $($c.CommandLine)"
  "----- screen -----"
  try { [ConReader2]::ReadScreen([uint32]$c.ProcessId) } catch { "ERROR: $($_.Exception.Message)" }
  ""
}
$out | Out-File -Encoding utf8 "read-ps-screen-out.txt"
