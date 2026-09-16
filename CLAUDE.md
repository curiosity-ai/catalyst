# Catalyst

C# natural language processing library (tokenization, tagging, entity recognition, embeddings).

## Build & test

```bash
dotnet build Catalyst/Catalyst.csproj -c Release
dotnet test --project tests/Catalyst.Tests/Catalyst.Tests.csproj -c Release
```

The tests are xunit.v3, which runs on Microsoft.Testing.Platform rather than VSTest. The
repo's `global.json` opts `dotnet test` into that runner; the .NET 10 SDK then wants the
project passed as `--project`, and `dotnet test <csproj>` is rejected.

## Git LFS required for tests (model files)

The `.bin` / `.binz` model files under `Languages/` and `Languages.ForTest/` are stored
with **Git LFS** (see `.gitattributes`). A plain clone that hasn't fetched LFS objects
leaves these as small text *pointer* files instead of the real binaries.

When that happens, tests that load a language model (anything going through
`English.Register()` / `Pipeline.ForAsync(..., tagger: true)`) fail while deserializing,
with an error like:

```
MessagePack.MessagePackSerializationException: Failed to deserialize
Catalyst.Models.AveragePerceptronTaggerModel value.
---- Unexpected msgpack code 118 (positive fixint) encountered.
```

`118` is `0x76` = `'v'`, the first byte of the LFS pointer text (`version https://git-lfs...`) —
the deserializer is reading the pointer instead of the model.

Fix by hydrating the LFS objects:

```bash
git lfs install --local
git lfs pull            # or scope it, e.g. --include="Languages.ForTest/English.ForTests/Resources/*"
```

Fresh/ephemeral environments (e.g. Claude Code on the web) clone without LFS content unless the
environment is set up to fetch it, so make sure LFS is configured there (or run `git lfs pull`
once per session) before running model-dependent tests.
