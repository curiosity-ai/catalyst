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

## Date and time recognition

`Catalyst/src/DateTime/` is a hand-written, regular-expression-free date/time engine that replaced the
`Microsoft.Recognizers.Text.DateTime` package. Do not reintroduce that dependency, and do not reach for
`System.Text.RegularExpressions` inside the engine — the whole point is a span scanner plus a lexeme grammar.
`docs/datetime.md` explains the pipeline and what each layer owns.

Two things are load-bearing and have tests:

- **The scan allocates nothing until something matches.** Buffers are rented, the lexicon is probed by
  `ReadOnlySpan<char>` through a `FrozenDictionary` alternate lookup, and the `Resolver` is created on the
  first hit. `AllocationTests` measures it.
- **Capability is measured, not asserted.** `tests/Catalyst.DateTime.Tests` scores the engine against the
  Microsoft.Recognizers.Text specification suite and against that library itself, per language. The floors in
  `ParityTests` catch regressions; raise one when the engine beats it, never lower one to make a change pass.
  That test project is the only place the old package is still referenced.

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

## Spotter segments

`Spotter` and `LinkedSpotter` keep their entries in `SpotterEngine` as a short list of immutable
`EntrySegment`s (a sorted front-coded `EntryDictionary`, its prefilter, the values by rank and a removal
bitmap). `AddEntry` / `RemoveEntry` only buffer; `Flush()` turns the buffer into a new segment and merges
segments by size (`SMALL_SEGMENT_ENTRIES`, `MERGE_FACTOR`), the way an inverted index merges its segments.
The newest segment holding an entry decides it, and only a merge into the oldest segment drops removals.

Load-bearing, with tests in `SpotterLiveMergeTests`:

- **Adding a few entries to a large model must not rewrite it.** Never reintroduce a re-open/re-freeze of the
  whole dictionary on `AddEntry`; `LinkedSpotter_AddingAFewEntriesDoesNotReallocateALargeModel` measures it.
- **Readers never wait on a writer, and never publish half a batch.** Recognition calls `FlushIfIdle`, which
  flushes only when no writer holds the lock and nobody has called `Flush()` explicitly; once the owner has,
  it alone decides when a batch (a removal and the addition replacing it) becomes visible.
- **The tokenizer-exception set is filled in place.** `FastTokenizer.ImportSpecialCases` registers a model's
  `CompactHash32Set` even while it is empty, and `CompactHash32Set.Add` keeps new hashes in a small overflow
  until it holds `OVERFLOW_LIMIT` of them.
