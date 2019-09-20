set mypath=%cd%
echo %mypath% 


"%mypath%\build\Release\cis565_stream_compaction_test.exe" -b 128 -n 1600000
timeout 60