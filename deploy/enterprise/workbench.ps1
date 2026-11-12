param(
  [ValidateSet('start','stop','status')][string]$Action='start',
  [ValidateSet('synthetic','public','corpus','fixed','contextual')][string]$Mode='synthetic',
  [ValidateSet('none','legacy','vllm')][string]$Model='none',
  [switch]$DirectModel,
  [int]$Port=8768,
  [string]$Python='D:\soft\Miniconda3_py313\envs\edu_agent\python.exe'
)
$ErrorActionPreference='Stop'
$projectRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$runtimePath=Join-Path $projectRoot 'runtime/enterprise'
New-Item -ItemType Directory -Force -Path $runtimePath | Out-Null
$recordPath=Join-Path $runtimePath "workbench-$Port.json"
$record=$null
if(Test-Path -LiteralPath $recordPath){$record=Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json}
$owned=$null
if($record){
  $candidate=Get-CimInstance Win32_Process -Filter "ProcessId=$($record.process_id)"
  if($candidate -and $candidate.CommandLine -and $candidate.CreationDate -and $candidate.CommandLine.Contains('-m rag_assist.enterprise serve') -and
     $candidate.CommandLine.Contains("--port $Port") -and
     $candidate.CreationDate.ToUniversalTime().Ticks -eq ([datetime]$record.creation_date).ToUniversalTime().Ticks){$owned=$candidate}
}
if($Action -eq 'stop'){
  if($owned){
    $termination=Invoke-CimMethod -InputObject $owned -MethodName Terminate
    if($termination.ReturnValue -ne 0){throw "Owned process termination failed: $($termination.ReturnValue)"}
    "Stopped project workbench on $Port."
  }else{"No owned process on $Port."}
  exit
}
if($Action -eq 'status'){
  @{running=[bool]$owned;port=$Port;mode=$record.mode;model=$record.model}|ConvertTo-Json
  exit
}
if($owned){"Workbench already running: http://127.0.0.1:$Port";exit}
if(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue){throw "Port $Port is occupied; no process was stopped."}
$arguments=@('-B','-m','rag_assist.enterprise','serve','--mode',$Mode,'--model',$Model,'--port',"$Port")
$previousDirect=$env:CLAIMMATE_LLM_DIRECT
try{
  if($DirectModel){$env:CLAIMMATE_LLM_DIRECT='1'}
  $process=Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runtimePath "workbench-$Port.log") -RedirectStandardError (Join-Path $runtimePath "workbench-$Port.err")
}finally{$env:CLAIMMATE_LLM_DIRECT=$previousDirect}
$actual=Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
@{process_id=$process.Id;creation_date=$actual.CreationDate.ToUniversalTime().ToString('o');mode=$Mode;model=$Model;port=$Port;direct_model=[bool]$DirectModel}|
  ConvertTo-Json|Set-Content -LiteralPath $recordPath -Encoding utf8
"Workbench dispatched: http://127.0.0.1:$Port ; token file: runtime/enterprise/access-token.txt"
