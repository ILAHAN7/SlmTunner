# SlmTunner Docker GPU Training Script
# Run from the project root directory

$ProjectRoot = $PSScriptRoot

Write-Host ">> Building SlmTunner Docker image..."
docker build -t slmtunner-gpu $ProjectRoot

Write-Host ">> Running container with GPU support..."
docker run --gpus all --memory=32g -it --rm `
    -v "${ProjectRoot}:/workspace" `
    --env-file "${ProjectRoot}/.env" `
    slmtunner-gpu /bin/bash
