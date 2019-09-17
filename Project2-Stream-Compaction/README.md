CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Joshua Nadel
  * https://www.linkedin.com/in/joshua-nadel-379382136/, http://www.joshnadel.com/
* Tested on: Windows 10, i7-6700HQ @ 2.60GHz 16GB, GTX 970M (Personal laptop)

### Scan / Stream Compaction

This project demonstrates the ability of the GPU to quickly complete algorithms that are, in serial, quite slow. It contains both serial and parallel implementations of scan and stream compaction algorithms, and uses timers to compare their performances on large quantities of data.
The list of features includes:
* Serial scan implementation on the CPU
* Serial compact implementation on the CPU
* Naive scan implementation on the GPU
* Work-efficient scan implementation on the GPU
* Work-efficient compact implementation on the GPU

![](img/timeOverLength.png)

I do not know how the thrust implementation manages to be so efficient at such large array sizes. In fact, it seems to become increasingly efficient with array length.

It is interesting to note that the efficient implementation is consistently slower than the naive implementation. This is likely due to memory access order.

Predictably, the serial CPU version increases in runtime proportionally to the increase in array size.

The program output reads:
```'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Users\Josh\Documents\School\UPenn\2019-2020\CIS 565\Project2-Number-Algorithms\Project2-Stream-Compaction\build\Release\cis565_stream_compaction_test.exe'. Module was built without symbols.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\ntdll.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\kernel32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\KernelBase.dll'. Symbols loaded.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\ucrtbase.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\msvcp140.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\vcruntime140.dll'. Symbols loaded.
The thread 0x4ab4 has exited with code 0 (0x0).
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\advapi32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\msvcrt.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\sechost.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\rpcrt4.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\gdi32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\gdi32full.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\msvcp_win.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\user32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\win32u.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\imm32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\setupapi.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\cfgmgr32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\bcrypt.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\devobj.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\wintrust.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\msasn1.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\crypt32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\shell32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\SHCore.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\combase.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\bcryptprimitives.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\windows.storage.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\profapi.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\powrprof.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\shlwapi.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\kernel.appcore.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\cryptsp.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\nvcuda.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\version.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\nvfatbinaryLoader.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\ws2_32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\dwmapi.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\uxtheme.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Unloaded 'C:\Windows\System32\dwmapi.dll'
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\nvapi64.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\ole32.dll'. Cannot find or open the PDB file.
'cis565_stream_compaction_test.exe' (Win32): Loaded 'C:\Windows\System32\dxgi.dll'. Cannot find or open the PDB file.
The thread 0x61a0 has exited with code 0 (0x0).
The thread 0x3554 has exited with code 0 (0x0).
The thread 0x3f60 has exited with code 0 (0x0).
The thread 0x3ef0 has exited with code 0 (0x0).
The thread 0x19f4 has exited with code 0 (0x0).
The thread 0x4948 has exited with code 0 (0x0).
The program '[11188] cis565_stream_compaction_test.exe' has exited with code 0 (0x0).```