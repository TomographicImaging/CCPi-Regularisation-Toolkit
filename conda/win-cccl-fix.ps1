# work-around conda-forge::cuda-cccl win-64 packaging bug:
# copy Library\include\targets\x64\ to Library\lib\include{,\cccl}
# so {thrust,cub,libcudacxx}-header-search.cmake finds headers
$ErrorActionPreference = "Stop"
$src = Join-Path $env:PREFIX "Library\include\targets\x64"
$dst = Join-Path $env:PREFIX "Library\lib\include"
if (Test-Path $src) {
    foreach ($d in Get-ChildItem $src -Directory) {
        $t = Join-Path $dst $d.Name
        if (-not (Test-Path $t)) {
            Copy-Item -Recurse $d.FullName $t
            Write-Host "cccl-fix: copied $($d.Name) -> $t"
        }
    }
    $v12 = Join-Path $dst "thrust\version.h"
    $v13 = Join-Path $dst "cccl\thrust\version.h"
    if (-not (Test-Path $v12) -and -not (Test-Path $v13)) {
        throw "cccl-fix: thrust\version.h not found under $dst after copy"
    }
    Write-Host "cccl-fix: headers in place at $dst"
} else {
    Write-Host "cccl-fix: no $src (nothing to do)"
}
