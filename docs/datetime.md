# Date and time recognition

`Catalyst.DateTimeRecognition` finds dates, times, periods, durations and recurrences in free text and
resolves them against a reference moment. It replaces the `Microsoft.Recognizers.Text.DateTime` dependency
Catalyst used to carry, and produces the same `datetimeV2` resolution shape, so anything reading the old
`timex` / `type` / `value` / `start` / `end` / `Mod` keys keeps working.

```csharp
var model = DateTimeModel.For(Language.English);

foreach (var hit in model.Parse("Let's meet next friday at 3pm", DateTime.Now))
{
    Console.WriteLine($"{hit.TypeName} [{hit.Start}..{hit.End}] {hit.Text}");

    foreach (var value in hit.Values)
    {
        Console.WriteLine($"   timex={value.Timex} value={value.Value} start={value.Start} end={value.End}");
    }
}
```

The entity recognizer used inside a pipeline is unchanged:

```csharp
var pipeline = Pipeline.TokenizerFor(Language.English);
pipeline.Add(new DateTimeRecognizer(Language.English));
pipeline.ProcessSingle(document);
```

Each recognised expression tags the tokens it covers with a `DateTime` entity whose `Metadata` is the first
resolution, as a `Dictionary<string, string>`.

## How it works

There is no regular expression anywhere in the engine. The pipeline is:

1. **`Lexer`** scans the `ReadOnlySpan<char>` into `Lexeme` structs — numbers (with their digit count), words,
   and single punctuation marks. It splits on script changes, so `3pm` and `mar3` become two lexemes the
   grammar can glue back together, and it records whether whitespace preceded each one.
2. **`Lexicon`** resolves each word to a `TermInfo` — a `TermKind` (month, weekday, unit, relative, part of
   day, modifier, holiday, …) plus a payload. Lookup goes through a `FrozenDictionary` alternate lookup keyed
   by `ReadOnlySpan<char>`, so a word is matched without ever being materialised as a string. Multi-word
   phrases ("new year's eve", "the day after tomorrow") are bucketed by first word and folded in a second pass.
3. **`Parser`** is a `ref partial struct` of hand-written matchers over the lexeme array. At each position it
   tries every construct and keeps the longest; ties go to the more specific reading. Nodes live in a
   caller-owned arena and reference each other by index, so a range holds its two endpoints without allocating.
4. **`Resolver`** turns a node into concrete dates and TIMEX strings against the reference moment, emitting the
   several readings an ambiguous expression has (a clock with no am/pm, a date with no year, a bare weekday).

Both buffers are rented from `ArrayPool`, and the `Resolver` is only created once something matches, so a scan
over text that contains no date allocates nothing at all. `AllocationTests` measures this rather than assuming it.

## What it recognises

Per type, with the TIMEX it produces:

| Type | Examples | TIMEX |
|---|---|---|
| `date` | `2019-08-01`, `jan 5`, `next friday`, `3 days ago`, `christmas`, `the 18th` | `2019-08-01`, `XXXX-01-05`, `XXXX-WXX-5` |
| `time` | `3pm`, `15:30`, `7:56:30 am`, `half past seven`, `noon` | `T15`, `T15:30`, `T07:56:30` |
| `datetime` | `tomorrow at 8:45`, `wed oct 26 15:50:06 2016`, `in 5 minutes`, `now` | `2016-11-08T08:45`, `PRESENT_REF` |
| `daterange` | `2019`, `april 2017`, `last week`, `q1 2019`, `from 2014 to 2018`, `1990s`, `week 23` | `2017-04`, `2018-W11`, `(2014-01-01,2018-01-01,P4Y)` |
| `timerange` | `morning`, `5 to 6pm`, `after 3pm`, `for 2 hours from 2pm` | `TMO`, `(T17,T18,PT1H)` |
| `datetimerange` | `tomorrow morning`, `monday 8-9am`, `tonight`, `next hour` | `2016-11-08TMO` |
| `duration` | `3 days`, `2w`, `one and a half hours`, `a few minutes` | `P3D`, `PT1.5H` |
| `set` | `every monday`, `weekly`, `tuesdays at 9am`, `19th of every month` | `XXXX-WXX-1`, `P1W` |

## Performance

Measured with BenchmarkDotNet (.NET 10, Intel Xeon 2.80GHz) against `Microsoft.Recognizers.Text.DateTime`
1.8.13 on the same inputs, both engines warmed up first so neither pays for building its patterns:

| Input | Microsoft | Catalyst | Faster | Microsoft allocated | Catalyst allocated |
|---|---:|---:|---:|---:|---:|
| Prose, 241 chars, no date in it | 972 µs | 138 µs | 7x | 52 KB | **0 B** |
| Short sentence, one date | 482 µs | 15 µs | 32x | 70 KB | 472 B |
| Sentence dense in date expressions | 2,782 µs | 45 µs | 62x | 290 KB | 3 KB |
| Document, ~9.5 KB, 240 hits | 636 ms | 5.2 ms | 122x | 43.5 MB | 105 KB |

The zero in the first row is the one that matters for a corpus: text with no date in it is what a scanner
spends nearly all of its time on, and there the engine allocates nothing at all.

## Languages

English is first class. German, French, Spanish, Portuguese, Italian and Dutch share the same grammar with
their own vocabulary; the grammar is written against `TermKind`, not against English words, and the handful of
genuinely language-shaped decisions (day-month order, decimal comma, a qualifier that follows its unit as in
*la semaine prochaine*) are flags on the `Lexicon`.

`Lexicons.IsSupported(language)` says whether a vocabulary exists; `DateTimeRecognizer` throws
`NotSupportedException` for anything else, so a caller can fall back to English.

## Parity with Microsoft.Recognizers.Text

`tests/Catalyst.DateTime.Tests` runs both engines over the specification suite from the
[Recognizers-Text](https://github.com/microsoft/Recognizers-Text) repository (`Specs/DateTime/<language>/DateTimeModel.json`,
MIT licensed, copied under `Specs/`) and scores them. Cases the suite itself marks as unsupported on .NET are
excluded, since the reference implementation does not meet them either — on what remains it scores 100%, which
is the ceiling this is measured against.

| Language | span + type | full resolution |
|---|---:|---:|
| English | 93.8% | 89.0% |
| EnglishOthers | 97.6% | 82.9% |
| Italian | 61.3% | 56.9% |
| French | 60.5% | 57.9% |
| Dutch | 49.7% | 44.4% |
| German | 43.0% | 37.6% |
| Spanish | 42.6% | 38.9% |
| Portuguese | 40.0% | 37.0% |

Adding a language, or improving one, is a matter of extending its lexicon and re-running the parity report; the
per-language floors in `ParityTests` exist to catch a regression, and should be raised whenever the engine
beats them.

A note on articles, because it is the one place the languages genuinely diverge. Whether a leading definite
article belongs to the match depends on what is being matched, not only on the language: English keeps it on a
date ("the 09th of may") and drops it from a qualified period ("the april 2017" is reported as "april 2017"),
while French, Spanish, Portuguese, Italian and Dutch do the opposite. That is what `Lexicon.ArticleInDateSpan`
and `ArticleInPeriodSpan` select between.

Two more per-language flags exist for the same reason. `RelativeAfterUnit` says whether the qualifier may follow
the unit (*la semaine prochaine*); English puts it in front, so reading it the other way round turns "2 hours
next month" into a two-hour period. `PluralEndsInS` says whether a plural unit can be told from a singular one
by its last letter, which is what makes "3 next week" the number three beside "next week" rather than three
weeks; German and Dutch opt out.
