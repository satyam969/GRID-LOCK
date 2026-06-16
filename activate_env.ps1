# TrafficGuard AI - Environment Activation Script
# Run this from the project root: .\activate_env.ps1

Write-Host "🚦 Activating TrafficGuard AI virtual environment..." -ForegroundColor Cyan

# Activate the venv
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

Write-Host "✅ Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host "  Start backend  : cd backend; uvicorn app.main:app --reload --port 8000" -ForegroundColor White
Write-Host "  Start frontend : cd frontend; npm run dev" -ForegroundColor White
Write-Host "  Run tests      : cd backend; python -m pytest tests/ -v" -ForegroundColor White
Write-Host "  Deactivate     : deactivate" -ForegroundColor White
