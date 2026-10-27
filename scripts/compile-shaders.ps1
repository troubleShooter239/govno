$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $shaderDirectory = 'crates/render-vulkan/shaders'
    # Build both stages in scratch, validate, then replace the checked-in binaries.
    $outputDirectory = 'target/shaders'
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    $compileTimes=[Collections.Generic.List[object]]::new()
    foreach ($entry in @('vertexMain', 'fragmentMain', 'postVertex', 'postFragment', 'shadowVertex', 'meterExposure', 'bloomExtract', 'bloomBlur', 'prefilterIBL', 'integrateBRDF', 'projectSH')) {
        $outputPath = Join-Path $outputDirectory "$entry.spv"
        $source = if ($entry -in @('prefilterIBL','integrateBRDF','projectSH')) { 'ibl.slang' } elseif ($entry -in @('meterExposure','bloomExtract','bloomBlur', 'prefilterIBL', 'integrateBRDF', 'projectSH')) { 'effects.slang' } elseif ($entry -eq 'shadowVertex') { 'shadow.slang' } elseif ($entry.StartsWith('post')) { 'post.slang' } else { 'forward.slang' }
        $timer=[Diagnostics.Stopwatch]::StartNew()
        slangc "$shaderDirectory/$source" -entry $entry -target spirv -profile spirv_1_6 -matrix-layout-column-major -fvk-use-entrypoint-name -reflection-json (Join-Path $outputDirectory "$entry.reflection.json") -o $outputPath
        $timer.Stop();$compileTimes.Add(@{entry=$entry;slang_process_ms=$timer.Elapsed.TotalMilliseconds})
        if ($LASTEXITCODE -ne 0) { throw "Slang compilation failed for $entry" }
        $reflectionPath=Join-Path $outputDirectory "$entry.reflection.json"
        $reflection=Get-Content $reflectionPath -Raw|ConvertFrom-Json
        $hash=[uint64]2166136261
        foreach($byte in [IO.File]::ReadAllBytes((Join-Path (Get-Location) $outputPath))){$hash=(($hash -bxor $byte)*[uint64]16777619) -band [uint64]4294967295}
        $reflection|Add-Member -NotePropertyName spirvFnv1a32 -NotePropertyValue ('{0:x8}' -f $hash)
        $reflection|ConvertTo-Json -Depth 100 -Compress|Set-Content $reflectionPath
        spirv-val --target-env vulkan1.3 $outputPath
        if ($LASTEXITCODE -ne 0) { throw "SPIR-V validation failed for $entry" }
    }
    foreach ($entry in @('vertexMain', 'fragmentMain', 'postVertex', 'postFragment', 'shadowVertex', 'meterExposure', 'bloomExtract', 'bloomBlur', 'prefilterIBL', 'integrateBRDF', 'projectSH')) {
        Copy-Item -LiteralPath (Join-Path $outputDirectory "$entry.reflection.json") -Destination (Join-Path $shaderDirectory "$entry.reflection.json") -Force
        Copy-Item -LiteralPath (Join-Path $outputDirectory "$entry.spv") -Destination (Join-Path $shaderDirectory "$entry.spv") -Force
    }
}
finally { if($compileTimes){$compileTimes|ConvertTo-Json -Depth 4|Set-Content (Join-Path $outputDirectory 'compile-times.json')};Pop-Location }
