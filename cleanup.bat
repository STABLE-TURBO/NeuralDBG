@echo off
REM Quick cleanup script for Windows users
REM Runs the PowerShell cleanup script with proper execution policy

echo ================================================================================
echo Neural DSL Repository Cleanup
echo ================================================================================
echo.
echo This will remove 200+ redundant files including:
echo   - 50+ implementation summary documents
echo   - 30+ quick reference duplicates
echo   - Duplicate workflows and .github docs
echo   - Temporary fix scripts
echo   - Test artifacts and cache files
echo.

:choice
set /P c=Do you want to preview changes first (dry-run)? [Y/N]:
if /I "%c%" EQU "Y" goto :dryrun
if /I "%c%" EQU "N" goto :execute
goto :choice

:dryrun
echo.
echo Running in DRY RUN mode (no files will be deleted)...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\cleanup_repository.ps1" -DryRun
echo.
echo.
set /P c2=Do you want to execute the cleanup now? [Y/N]:
if /I "%c2%" EQU "Y" goto :execute
if /I "%c2%" EQU "N" goto :end
goto :dryrun

:execute
echo.
echo Executing cleanup...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0scripts\cleanup_repository.ps1"
echo.
echo Done!
goto :end

:end
echo.
pause
