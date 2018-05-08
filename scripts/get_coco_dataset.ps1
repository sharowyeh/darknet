## Clone COCO API
git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

## Download Images
$wc = new-object System.Net.WebClient
$wc.DownloadFile("https://pjreddie.com/media/files/train2014.zip", "train2014.zip")
$wc.DownloadFile("https://pjreddie.com/media/files/val2014.zip", "val2014.zip")

## Unzip
## In PS5.0 uses native support can overwrite exists destination files
Expand-Archive -Path train2014.zip -DestinationPath .\ -Force
Expand-Archive -Path val2014.zip -DestinationPath .\ -Force
#[System.IO.Compression.ZipFile]::ExtractToDirectory("train2014.zip", ".\")
#[System.IO.Compression.ZipFile]::ExtractToDirectory("val2014.zip", ".\")

cd ..

## Download COCO Metadata
$wc.DownloadFile("https://pjreddie.com/media/files/instances_train-val2014.zip", "instances_train-val2014.zip")
$wc.DownloadFile("https://pjreddie.com/media/files/coco/5k.part", "5k.part")
$wc.DownloadFile("https://pjreddie.com/media/files/coco/trainvalno5k.part", "trainvalno5k.part")
$wc.DownloadFile("https://pjreddie.com/media/files/coco/labels.tgz", "labels.tgz")

## Before extract tar gzip file, powershell needs nuget package provider for getting module from PowerShellGallery
## https://docs.microsoft.com/en-us/powershell/gallery/psgallery/psgallery_gettingstarted
## If powershell does not have NuGet package provider, it will install nuget library at:
## C:\Users\<username>\AppData\Local\PackageManagement\ProviderAssemblies
Find-PackageProvider -Name NuGet -Force

## Import 7Zip module for powershell to extract gzip tar file
## https://www.powershellgallery.com/packages/7Zip4PowerShell/
if (-Not (Get-Command Expand-7Zip -ErrorAction Ignore)) {
    Save-Module -Name 7Zip4Powershell -Path $env:LOCALAPPDATA -RequiredVersion 1.8.0
    Import-Module $env:LOCALAPPDATA\7Zip4Powershell\1.8.0\7Zip4PowerShell.psd1
}
Write-Host "Get-Module 7Zip4PowerShell"
Get-Module -Name 7Zip4PowerShell

Write-Host "Expand-7Zip labels.tgz ..."
Expand-7Zip labels.tgz .
Expand-7Zip labels.tar .

Write-Host "Expand-Archive ..."
Expand-Archive -Path instances_train-val2014.zip -DestinationPath .\ -Force
#[System.IO.Compression.ZipFile]::ExtractToDirectory("instances_train-val2014.zip", ".\")

## Set Up Image Lists
## Using System.IO.Stream prevent performance issue
$sw = new-object System.IO.StreamWriter($PWD.ToString()+"\5k.txt")
foreach($line in [System.IO.File]::ReadLines($PWD.ToString()+"\5k.part")) {
    $sw.write($PWD.ToString()+$line+"`n")
}
$sw.close()
Write-Host "Export 5k.txt"
$sw = new-object System.IO.StreamWriter($PWD.ToString()+"\trainvalno5k.txt")
foreach($line in [System.IO.File]::ReadLines($PWD.ToString()+"\trainvalno5k.part")) {
    $sw.write($PWD.Tostring()+$line+"`n")
}
$sw.close()
Write-Host "Export trainvalno5k.txt"

