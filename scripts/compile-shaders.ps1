$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $shaderDirectory = 'crates/render-vulkan/shaders'
    # Build both stages in scratch, validate, then replace the checked-in binaries.
    $outputDirectory = 'target/shaders'
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    foreach ($entry in @('vertexMain', 'fragmentMain', 'postVertex', 'postFragment', 'shadowVertex', 'meterExposure', 'bloomExtract', 'bloomBlur', 'prefilterIBL', 'integrateBRDF', 'projectSH', 'vertexMainBindless', 'fragmentMainBindless')) {
        [string[]]$defines = @()
        if ($entry.EndsWith('Bindless')) { $defines = @('-DGOVNO_BINDLESS=1') }
        $shaderEntry = $entry -replace 'Bindless$',''
        $outputPath = Join-Path $outputDirectory "$entry.spv"
        $source = if ($shaderEntry -in @('prefilterIBL','integrateBRDF','projectSH')) { 'ibl.slang' } elseif ($shaderEntry -in @('meterExposure','bloomExtract','bloomBlur')) { 'effects.slang' } elseif ($shaderEntry -eq 'shadowVertex') { 'shadow.slang' } elseif ($shaderEntry.StartsWith('post')) { 'post.slang' } else { 'forward.slang' }
        slangc "$shaderDirectory/$source" -entry $shaderEntry @defines -target spirv -profile spirv_1_6 -matrix-layout-column-major -fvk-use-entrypoint-name -reflection-json (Join-Path $outputDirectory "$entry.reflection.json") -o $outputPath
        if ($LASTEXITCODE -ne 0) { throw "Slang compilation failed for $entry" }
        $reflectionPath=Join-Path $outputDirectory "$entry.reflection.json"
        $reflection=Get-Content $reflectionPath -Raw|ConvertFrom-Json
        $hash=[uint64]2166136261
        foreach($byte in [IO.File]::ReadAllBytes((Join-Path (Get-Location) $outputPath))){$hash=(($hash -bxor $byte)*[uint64]16777619) -band [uint64]4294967295}
        $reflection|Add-Member -NotePropertyName spirvFnv1a32 -NotePropertyValue ('{0:x8}' -f $hash)
        $reflectionJson = $reflection | ConvertTo-Json -Depth 100 -Compress
        [IO.File]::WriteAllText((Join-Path (Get-Location) $reflectionPath), $reflectionJson + "`n", [Text.UTF8Encoding]::new($false))
        spirv-val --target-env vulkan1.4 $outputPath
        if ($LASTEXITCODE -ne 0) { throw "SPIR-V validation failed for $entry" }
    }
    foreach ($entry in @('vertexMain', 'fragmentMain', 'postVertex', 'postFragment', 'shadowVertex', 'meterExposure', 'bloomExtract', 'bloomBlur', 'prefilterIBL', 'integrateBRDF', 'projectSH', 'vertexMainBindless', 'fragmentMainBindless')) {
        Copy-Item -LiteralPath (Join-Path $outputDirectory "$entry.reflection.json") -Destination (Join-Path $shaderDirectory "$entry.reflection.json") -Force
        Copy-Item -LiteralPath (Join-Path $outputDirectory "$entry.spv") -Destination (Join-Path $shaderDirectory "$entry.spv") -Force
    }
}
finally { Pop-Location }
