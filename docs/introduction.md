# Introduction to Catalyst

**Catalyst** is a high-performance Natural Language Processing (NLP) library for C#. Inspired by the design of [spaCy](https://spacy.io/), Catalyst provides a fast and modern way to process text in .NET.

## Key Features

- **Fast & Modern**: Built in pure C#, supporting .NET Standard 2.0 and later.
- **Cross-Platform**: Runs on Windows, Linux, macOS, and ARM.
- **Efficient Tokenization**: Non-destructive, RegEx-free tokenization capable of processing over 1 million tokens per second.
- **Comprehensive NLP Tasks**: Support for Tokenization, Sentence Detection, Part-of-Speech (POS) Tagging, Named Entity Recognition (NER), Language Detection, and more.
- **Pre-trained Models**: Easy access to pre-trained models for various languages.

## Core Concepts

Understanding Catalyst requires familiarity with its core building blocks:

### 1. Document
The `Document` class represents a piece of text to be processed. It holds the original text and all the metadata generated during processing (tokens, spans, entities, etc.).

```csharp
var doc = new Document("The quick brown fox jumps over the lazy dog", Language.English);
```

### 2. Span
A `Span` represents a segment of a `Document`, typically a sentence. A `Document` can contain multiple spans.

### 3. Token
A `Token` is the basic unit of text, such as a word or punctuation mark, within a `Span`.

### 4. Pipeline
A `Pipeline` is a sequence of processing steps (models) that are applied to a `Document`. A typical pipeline includes a tokenizer, a sentence detector, and a POS tagger.

```csharp
var nlp = await Pipeline.ForAsync(Language.English);
nlp.ProcessSingle(doc);
```

### 5. Language
Catalyst uses the `Language` enum to specify the language of a document or model. It supports a wide range of languages.

## Getting Started

To use Catalyst, you need to install the `Catalyst` NuGet package.

### Language Packages

All language-specific data and models are provided as separate NuGet packages. You can find all available packages [here](https://www.nuget.org/packages?q=catalyst.models).

Before using a language, you must install its respective NuGet package and register it in your code.

```bash
# Example: Adding English language support via dotnet CLI
dotnet add package Catalyst.Models.English
```

```csharp
using Catalyst;
using Catalyst.Models;
using Mosaik.Core;

// Register the English language models
Catalyst.Models.English.Register();

// Configure storage for lazy-loading models
Storage.Current = new DiskStorage("catalyst-models");

// Create a pipeline for English
var nlp = await Pipeline.ForAsync(Language.English);

// Create and process a document
var doc = new Document("Hello, world!", Language.English);
nlp.ProcessSingle(doc);

// Access the results
Console.WriteLine(doc.ToJson());
```

## Storage
Catalyst uses a storage mechanism to load and cache models. By default, it can download models from an online repository or load them from a local disk using `DiskStorage`.

```csharp
Storage.Current = new DiskStorage("catalyst-models");
```

## Supported Languages

Below is a list of supported languages and their corresponding NuGet packages.

| Language | NuGet Package |
| --- | --- |
| English | [Catalyst.Models.English](https://www.nuget.org/packages/Catalyst.Models.English) |
| French | [Catalyst.Models.French](https://www.nuget.org/packages/Catalyst.Models.French) |
| German | [Catalyst.Models.German](https://www.nuget.org/packages/Catalyst.Models.German) |
| Spanish | [Catalyst.Models.Spanish](https://www.nuget.org/packages/Catalyst.Models.Spanish) |
| Italian | [Catalyst.Models.Italian](https://www.nuget.org/packages/Catalyst.Models.Italian) |
| Dutch | [Catalyst.Models.Dutch](https://www.nuget.org/packages/Catalyst.Models.Dutch) |
| Portuguese | [Catalyst.Models.Portuguese](https://www.nuget.org/packages/Catalyst.Models.Portuguese) |
| Russian | [Catalyst.Models.Russian](https://www.nuget.org/packages/Catalyst.Models.Russian) |
| Chinese | [Catalyst.Models.Chinese](https://www.nuget.org/packages/Catalyst.Models.Chinese) |
| Japanese | [Catalyst.Models.Japanese](https://www.nuget.org/packages/Catalyst.Models.Japanese) |
| ... and many more | [See all packages](https://www.nuget.org/packages?q=catalyst.models) |

For a full list of supported languages, check the `Languages` folder in the repository or search NuGet for `Catalyst.Models`.
