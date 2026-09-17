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

Three rates, because a span disagreement and a wrong answer are not the same thing. **Same reading** is the
one that says whether the engine understood the text: the resolution matches field for field, and the span is
either identical or differs only by glue — an article, a preposition, a comma — which the reference
implementation is not consistent about itself, reporting *am Wochenende* with its preposition and *am Freitag*
without. The tolerance is narrow: it is reached only when the readings already match, and every word in the
disagreement has to be one the language's own lexicon classes as glue, so a content word still counts as a
miss. Those cases are listed as `GLUE` rather than `SPAN` in the report. The two strict rates are reported
beside it, so the tolerance hides nothing.

| Language | same reading | span + type | full resolution |
|---|---:|---:|---:|
| EnglishOthers | 95.1% | 97.6% | 95.1% |
| English | 94.6% | 96.0% | 93.9% |
| German | 93.7% | 93.2% | 91.0% |
| Italian | 93.4% | 92.0% | 89.1% |
| Portuguese | 92.7% | 90.3% | 89.1% |
| French | 91.1% | 88.4% | 86.3% |
| Dutch | 90.6% | 86.3% | 83.5% |
| Spanish | 90.3% | 85.3% | 82.9% |

Adding a language, or improving one, is a matter of extending its lexicon and re-running the parity report; the
per-language floors in `ParityTests` exist to catch a regression, and should be raised whenever the engine
beats them.

A note on articles, because it is the one place the languages genuinely diverge. Whether a leading definite
article belongs to the match depends on what is being matched, not only on the language: English keeps it on a
date ("the 09th of may") and drops it from a qualified period ("the april 2017" is reported as "april 2017"),
while French, Spanish, Portuguese, Italian, Dutch and German do the opposite. That is what
`Lexicon.ArticleInDateSpan` and `ArticleInPeriodSpan` select between.

The other per-language flags exist for the same reason — a rule that would otherwise have an English word
written into it:

- `RelativeAfterUnit` — whether the qualifier may follow the unit (*la semaine prochaine*). English puts it in
  front, so reading it the other way round turns "2 hours next month" into a two-hour period.
- `PluralEndsInS` — whether a plural unit can be told from a singular one by its last letter, which is what
  makes "3 next week" the number three beside "next week" rather than three weeks. German and Dutch opt out,
  and Dutch needs to: it writes a single morning as *'s morgens*.
- `PartNamedWithOf` — whether naming part of a period takes a preposition ("the end of may"). Where it does, a
  bare "start" or "end" in front of anything else is the verb; where it does not (*Anfang Mai*) it is not.
- `MinutesFollowHour` — whether the minutes are spoken after the hour and joined to it (*siete y media*).
- `HalfIsBeforeTheHour` — whether "half" names the half hour *before* the hour it precedes, so *halb acht* and
  *half acht* are half past seven. The same reading covers the quarters: *viertel acht*, *dreiviertel acht*.
- `SplitsCompounds` — whether the language writes compounds as one word (*dienstagmorgen*, *neunundzwanzig*),
  so an unknown word is worth splitting into two the lexicon does know.
- `OrdinalEndsInDot` — whether an ordinal is written as its number and a full stop (*22. April*).
- `MovableHolidayNamesItsDay` — whether a feast that falls on a different day each year still names a day in
  its timex. The suites disagree: English reports *easter monday* as `XXXX-04-22`, German reports
  *Ostermontag* as `XXXX`.
- `DecimalComma`, `DayMonthOrder` — how a number and a numeric date are written.

Two splits the scanner performs need no flag, because both halves have to be known words for them to happen at
all: a word elided onto the next one (*un'ora*) is split at its apostrophe, and the tens and units run together
(*ventinove*, *veinticuatro*) are read as one number.

Four term kinds carry a distinction the flags cannot:

- `TermKind.Article` marks the glue that can *head* a phrase, as against a preposition that cannot — which is
  what makes *por una hora* report from the article while *den ganzen Tag* keeps its.
- `TermKind.Whole` marks the word for "whole", which is what lets that phrase count as one.
- `TermKind.AndWord` marks the language's "and". It closes a range that *between* opened and cannot open one
  by itself, so *después de 2016 y antes de 2018* stays two ranges.
- `TermKind.ClockPrefix` marks what introduces a reading — *at*, *a las*, *um*, *à*. It licenses a bare hour,
  and an hour that follows one is a clock rather than a count of hours.

A word may hold two roles at once (`TermInfo` carries a kind and an alternate), and that is how most of this is
expressed: the Romance *de* is glue *and* a range opener, the German *nachmittags* is a part of the day *and*
an am/pm marker, the French *à* joins a range *and* introduces a clock. Registering the same word twice does
not add a role — the later entry replaces the earlier one, silently, which has been the single most common way
for a language to lose a reading it looked like it had.
