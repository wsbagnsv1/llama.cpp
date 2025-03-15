@echo off
setlocal enabledelayedexpansion

echo ===== Ovis2-1B to GGUF Conversion Script =====

:: Define temporary directory for the model
set "MODEL_TEMP_DIR=%CD%\temp_ovis2_1b"
echo Using temporary directory: %MODEL_TEMP_DIR%

:: Create temporary directory if it doesn't exist (do not delete if it already exists)
if not exist "%MODEL_TEMP_DIR%" (
    mkdir "%MODEL_TEMP_DIR%"
) else (
    echo Temporary directory already exists.
)

:: Define output directory for the converted model (changed to ovis2\ggufs)
set "OUTPUT_DIR=f:\Users\timbe\Desktop\llama.cpp-master\models\ovis2\ggufs"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Set default quantization if not provided, else use the first argument
if "%1"=="" (
    set "QUANT=f16"
) else (
    set "QUANT=%1"
)
echo Original quantization parameter: %QUANT%

:: Convert quantization argument to lowercase using PowerShell
for /f "usebackq delims=" %%x in (`powershell -NoProfile -Command "$env:QUANT.ToLower()"`) do set "QUANT=%%x"
echo Using quantization: %QUANT%

:: Disable Git clone protection
set GIT_CLONE_PROTECTION_ACTIVE=false

:: Check if the model repository is already cloned (using README.md as a marker)
if exist "%MODEL_TEMP_DIR%\README.md" (
    echo Model already downloaded, skipping git clone.
) else (
    echo Downloading Ovis2-1B model from Huggingface...
    git lfs install
    cd "%MODEL_TEMP_DIR%"
    git clone https://huggingface.co/AIDC-AI/Ovis2-1B .
    if %ERRORLEVEL% neq 0 (
        echo Failed to download the model.
        goto :end
    )
)

:: Check if the converted model already exists (includes quantization in the filename)
if exist "%OUTPUT_DIR%\ovis2-1b_%QUANT%.gguf" (
    echo Converted model already exists, skipping conversion.
    goto :end
)

:: Convert the model to GGUF format with the specified quantization
echo Converting model to GGUF format with quantization %QUANT%...
cd "f:\Users\timbe\Desktop\llama.cpp-master"
python convert_hf_to_gguf.py --outtype %QUANT% --outfile "%OUTPUT_DIR%\ovis2-1b_%QUANT%.gguf" "%MODEL_TEMP_DIR%"
if %ERRORLEVEL% neq 0 (
    echo Failed to convert the model.
    goto :end
)

echo Conversion completed successfully!
echo GGUF model saved to: %OUTPUT_DIR%\ovis2-1b_%QUANT%.gguf

:end
echo Process finished.
endlocal
exit /b
