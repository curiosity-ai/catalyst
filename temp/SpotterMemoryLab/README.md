# Spotter memory lab

Throwaway project measuring the memory footprint of `Spotter` / `LinkedSpotter` at the scale we
actually use them (10M aviation part numbers), and of several alternative designs.

Everything reported is an **analytic size model** (`src/Sz.cs`): counted array elements and object
fields on 64-bit, never process working set or GC counters. The model is cross-checked against the
real Catalyst models built over the same dataset - `Spotter.OptimizedMemoryBytes` and the real
`Dictionary`/`HashSet` capacities read back via `EnsureCapacity(0)`. Model and measurement agree to
0.3%.

## Running it

```bash
dotnet build -c Release
dotnet run -c Release --no-build -- --n 10000000            # structured catalogue
dotnet run -c Release --no-build -- --n 10000000 --flat --no-real   # low-structure bracket
```

`--n` sets the entry count, `--flat` removes almost all family structure (pessimistic bound for the
structure-dependent designs), `--no-real` skips building the real Catalyst models.

Captured output is in `results/`. The whole 10M run takes ~80s and peaks well under 4 GB.

## What is in here

| file | what it is |
| --- | --- |
| `src/PartNumberGenerator.cs` | synthetic aviation catalogue: AN/MS/NAS/NASM standards, Boeing BAC and numeric drawings, Airbus ASNA/NSA/ABS, MIL connectors, Hi-Lok/Hi-Shear/Cherry, OEM sequential, bearings, material specs, vendor free-form - organised into families that stock a random subset of a shared dash-number pool |
| `src/Sz.cs` | the size model, including the BCL's prime-doubling capacity chain |
| `src/BaselineModel.cs` | analytic sizes of today's structures, from `CompactHashStructures.cs` / `MphPerfectHash.cs` |
| `src/EliasFano.cs` | Elias-Fano coded sorted hashes |
| `src/FrontCodedDictionary.cs` | prefix-compressed sorted string dictionary, rank-addressable |
| `src/Dafsa.cs` | minimal acyclic automaton (Daciuk incremental) with rank counters, i.e. a minimal perfect hash |
| `src/BlockedBloom.cs` | one-cache-line prefilter used in front of the exact designs |
| `src/NgramAnalysis.cs` | n-grams as an inverted index and as a compressor |

## Correctness

The automaton is verified by checking `Rank(word_i) == i` for every entry. Because `Rank` counts the
strings the automaton accepts lexicographically before the query, all ranks being exactly `0..n-1`
proves the accepted language is *exactly* the input set - it cannot have over-generalised. The
front-coded dictionary is checked the same way, and both are probed with 1M near-miss non-members
(real part numbers with one character changed) to confirm zero false positives.
