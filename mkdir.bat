@echo off
setlocal

set "IOX2_DIR=C:\Temp\iceoryx2"

if not exist "%IOX2_DIR%\" (
    echo Creating %IOX2_DIR% ...
    mkdir "%IOX2_DIR%"

    if errorlevel 1 (
        echo Failed to create %IOX2_DIR%
        echo Try running this .bat as Administrator, or choose another writable path.
        exit /b 1
    )
)

echo iceoryx2 directory is ready: %IOX2_DIR%

endlocal