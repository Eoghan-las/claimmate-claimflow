# Standalone demo deployment

The same synthetic application has been container-built and checked on Linux/amd64. The verification covers seven HTTP checks: health, authentication, source dependencies, multi-turn product reference, clarification, missing materials, and supplementation with a Decimal estimate of 640.01. See `container-verification.json`.

## Prebuilt GitHub Release asset

The preparation deliverables include `claimmate-test-build.zip`. Extract it and load the image:

```sh
docker load --input claimmate-demo-image.tar.gz
```

Run it in PowerShell:

```powershell
$env:CLAIMMATE_ACCESS_TOKEN = [guid]::NewGuid().ToString('N')
Write-Output ("Local demo token: " + $env:CLAIMMATE_ACCESS_TOKEN)
docker run --rm -p 127.0.0.1:8000:8000 -e CLAIMMATE_ACCESS_TOKEN claimmate-demo:submission-20261112
```

Copy the token displayed locally into the workbench at **http://127.0.0.1:8000**, then connect. Keep the token out of public screenshots and descriptions. A new token is generated for this local run; no credential is baked into the image.

On other operating systems, set `CLAIMMATE_ACCESS_TOKEN` in the launching shell to a private value, then use the same Docker run command. Use Docker's Linux/amd64 container support or build the source for your own target platform.

This command does not mount host folders. State remains inside the temporary container and is removed when that `--rm` container stops.

## Build from source

```sh
docker build -t claimmate-demo:submission-20261112 .
```

The Dockerfile copies the project's MIT license into the image. Installed third-party distributions retain their own license files.

## Hosted application

A public hosted application is a separate publication step. Use a host that can run the image, route HTTPS to container port 8000, and provide `CLAIMMATE_ACCESS_TOKEN` as a runtime secret. Provide the reviewer credential privately and repeat the same user journey through the actual hosted URL before publishing it.

The downloadable Release asset is a test build, and the local address above is not a public hosted URL. The application uses one shared synthetic demonstration workspace; it is not a production multi-tenant deployment.
