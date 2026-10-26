$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $shaderDirectory = 'crates/render-vulkan/shaders'
    # Build both stages in scratch, validate, then replace the checked-in binaries.
    $outputDirectory = 'target/shaders'
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
    foreach ($entry in @('vertexMain', 'fragmentMain', 'postVertex', 'postFragment', 'shadowVertex')) {
        $outputPath = Join-Path $outputDirectory "$entry.spv"
        $source = if ($entry -eq 'shadowVertex') { 'shadow.slang' } elseif ($entry.StartsWith('post')) { 'post.slang' } else { 'forward.slang' }
        slangc "$shaderDirectory/$source" -entry $entry -target spirv -profile spirv_1_6 -matrix-layout-column-major -fvk-use-entrypoint-name -o $outputPath
        if ($LASTEXITCODE -ne 0) { throw "Slang compilation failed for $entry" }
        spirv-val --target-env vulkan1.3 $outputPath
        if ($LASTEXITCODE -ne 0) { throw "SPIR-V validation failed for $entry" }
    }
    foreach ($entry in @('vertexMain', 'fragmentMain', 'postVertex', 'postFragment', 'shadowVertex')) {
        Copy-Item -LiteralPath (Join-Path $outputDirectory "$entry.spv") -Destination (Join-Path $shaderDirectory "$entry.spv") -Force
    }
}
finally { Pop-Location }
