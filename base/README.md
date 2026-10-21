# Base engine build files (dsv41-1017-ds41f:r6w)

These are the files that built this recipe's base image `dsv41-1017-ds41f:r6w`: `docker/Dockerfile`
and its eight patches, copied **unmodified** from
justincasey/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark at commit `fa77c26` (`docker/`), which is no longer
online. They are byte-identical to the build directory the measurement Spark used on 2026-10-18
(`SHA256SUMS` here). `LICENSE` (Apache-2.0) and `NOTICE` are that repository's, kept with the files;
`NOTICE` lists every third-party component the build fetches and its license. The engine itself is
Christopher Owen's [spark-ds41f](https://github.com/christopherowen/spark-ds41f) (Apache-2.0) r6w,
fetched at build time at commit `2dc95329`.

**Prebuilt (default).** `scripts/build_image.sh` does not build this image. When it is
missing, `scripts/fetch_base.sh` downloads the redistributable copy published at
[vcruz305/DSV4.1-Flash-1017-Spark-base-image](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-Spark-base-image)
(the source build of these files with NVIDIA CUTLASS DSL removed, flattened to one layer), checks it,
and reinstalls CUTLASS DSL 4.7.1 from PyPI with `Dockerfile.cutlass-dsl` and `cutlass-dsl.lock`
(sha256-pinned wheels). CUTLASS DSL's licence allows redistributing only its Python source, which is
why it is not in the published image. See the main README, "Base image".

**From source.** `BASE_FROM_SOURCE=1 bash scripts/build_image.sh`, or by hand.
Build (CPU only, no GPU needed; several hours, ~100+ GB of disk for the build, as much RAM as you can
give it), from the recipe root:

```bash
sudo docker build --platform linux/arm64 -f base/docker/Dockerfile -t dsv41-1017-ds41f:r6w base/
```

The build fetches pinned inputs from Docker Hub (the vLLM nightly image by digest), GitHub
(spark-ds41f, Local Inference Lab vLLM and B12X, TileLang, TileKernels, NCCL, CUTLASS,
dgx-spark-networking) and PyPI (hash-locked wheels). These files are kept as the source of record;
a rebuild on 2026-10-20 (Apple Silicon, Docker Desktop, 12 CPUs, 18.5 min) reproduced every
source file and every label of the 2026-10-18 build; only compiled binaries and build logs differ
(see the main README, "Base image").
