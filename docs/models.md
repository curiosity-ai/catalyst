# Models in Catalyst

Catalyst provides several types of models to handle different NLP tasks. This page explains the most common model types and how to use them.

## Tokenizers

Tokenizers split raw text into individual tokens (words, punctuation, etc.).

### FastTokenizer
The default tokenizer in Catalyst. It is highly efficient and non-destructive.

```csharp
var tokenizer = new FastTokenizer(Language.English);
```

## Part-of-Speech (POS) Taggers

POS taggers assign grammatical tags (e.g., Noun, Verb, Adjective) to each token. Catalyst uses [Universal Dependencies](https://universaldependencies.org/) for POS tagging.

### AveragePerceptronTagger
A fast and accurate POS tagger based on the averaged perceptron algorithm.

```csharp
var tagger = await AveragePerceptronTagger.FromStoreAsync(Language.English, Version.Latest, "");
```

**Example Output:**

For the sentence: *"The quick brown fox jumps over the lazy dog"*

| Token | POS Tag |
| --- | --- |
| The | DET |
| quick | ADJ |
| brown | ADJ |
| fox | NOUN |
| jumps | VERB |
| over | ADP |
| the | DET |
| lazy | ADJ |
| dog | NOUN |

## Named Entity Recognition (NER)

NER models identify and categorize entities in text (e.g., Names, Organizations, Locations). Catalyst supports three main types:

### 1. Spotter
A gazetteer-like model that matches a predefined set of words or phrases.

```csharp
var spotter = new Spotter(Language.Any, 0, "programming", "ProgrammingLanguage");
spotter.AddEntry("C#");
spotter.AddEntry("Python");
```

**Example Output:**

For the sentence: *"I love coding in C# and Python."*

| Entity | Entity Type |
| --- | --- |
| C# | ProgrammingLanguage |
| Python | ProgrammingLanguage |

### 2. PatternSpotter
A rule-based model that uses complex patterns of tokens to identify entities. Conceptual equivalent of RegEx but on tokens.

```csharp
var isApattern = new PatternSpotter(Language.English, 0, "is-a-pattern", "IsA");
isApattern.NewPattern(
    "Is+Noun",
    mp => mp.Add(
        new PatternUnit(P.Single().WithToken("is").WithPOS(PartOfSpeech.VERB)),
        new PatternUnit(P.Multiple().WithPOS(PartOfSpeech.NOUN, PartOfSpeech.PROPN, PartOfSpeech.DET, PartOfSpeech.ADJ))
));
```

**Example Output:**

For the sentence: *"Catalyst is a high-performance library."*

| Match | Entity Type |
| --- | --- |
| is a high-performance library | IsA |

### 3. AveragePerceptronEntityRecognizer
A statistical model for NER, typically trained on large datasets like WikiNER.

```csharp
var ner = await AveragePerceptronEntityRecognizer.FromStoreAsync(Language.English, Version.Latest, "WikiNER");
```

### Iterating through Entities

Once a document has been processed by an NER model, you can iterate through the captured entities:

```csharp
foreach (var span in doc)
{
    foreach (var entity in span.GetEntities())
    {
        Console.WriteLine($"Entity: {entity.Value} [{entity.EntityType.Type}]");
    }
}
```

Or using LINQ:

```csharp
var entities = doc.SelectMany(span => span.GetEntities());
foreach(var entity in entities)
{
    Console.WriteLine($"Entity: {entity.Value} [{entity.EntityType.Type}]");
}
```

## Embeddings

Embeddings represent words or documents as dense vectors in a continuous vector space.

### FastText
Supports training and using FastText word and document embeddings.

```csharp
var ft = new FastText(Language.English, 0, "my-fasttext-model");
ft.Train(nlp.Process(docs));
```

**Example: Vector Retrieval and Similarity**

```csharp
// Get vector for a word
float[] vector = ft.GetVector("apple", Language.English);

// Compute similarity between two words
float[] vector1 = ft.GetVector("apple", Language.English);
float[] vector2 = ft.GetVector("orange", Language.English);
float similarity = vector1.CosineSimilarityWith(vector2);
```

## Language Detectors

Language detectors identify the language of a given text.

### FastTextLanguageDetector
Uses FastText models for accurate language detection.

```csharp
var detector = await FastTextLanguageDetector.FromStoreAsync(Language.Any, Version.Latest, "");
```

### LanguageDetector
Derived from Google's CLD3 (Compact Language Detector 3).

```csharp
var detector = await LanguageDetector.FromStoreAsync(Language.Any, Version.Latest, "");
```

## Normalizers

Normalizers transform text to a standard form (e.g., lowercasing, removing punctuation).

| Normalizer | Input | Output |
| --- | --- | --- |
| `LowerCaseNormalizer` | *"Hello World"* | *"hello world"* |
| `UpperCaseNormalizer` | *"Hello World"* | *"HELLO WORLD"* |
| `HtmlNormalizer` | *"&lt;b&gt;Hello&lt;/b&gt;"* | *"Hello"* |
| `FoldToAsciiNormalizer` | *"Crème brûlée"* | *"Creme brulee"* |
| `RemovePunctuationNormalizer` | *"Hello, World!"* | *"Hello World"* |

**Usage:**

```csharp
nlp.Add(new LowerCaseNormalizer());
```
