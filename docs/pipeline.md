# Pipeline Usage

The `Pipeline` class is the central orchestrator in Catalyst. It defines the sequence of processing steps that are applied to documents.

## Creating a Pipeline

### Default Pipeline
You can easily create a default pipeline for a specific language using `Pipeline.ForAsync`. This typically includes a tokenizer, a sentence detector, and a POS tagger.

```csharp
var nlp = await Pipeline.ForAsync(Language.English);
```

### Tokenizer-only Pipeline
If you only need tokenization (and optionally sentence detection), use `Pipeline.TokenizerForAsync`.

```csharp
var nlp = await Pipeline.TokenizerForAsync(Language.English, sentenceDetector: true);
```

## Customizing the Pipeline

Pipelines are flexible and allow you to add or remove processing steps.

### Adding Processes
You can add any model that implements `IProcess` to the pipeline.

```csharp
var nlp = await Pipeline.ForAsync(Language.English);
nlp.Add(await AveragePerceptronEntityRecognizer.FromStoreAsync(Language.English, Version.Latest, "WikiNER"));
```

### Custom Order
When you add a process using `Add()`, Catalyst automatically maintains a logical order:
1. Normalizers
2. Tokenizers
3. Sentence Detectors
4. Taggers
5. Others (e.g., Entity Recognizers)

### Removing Processes
You can remove models from the pipeline if they are no longer needed.

```csharp
nlp.RemoveAll(p => p is ITagger);
```

## Processing Documents

### Single Document
For processing a single document, use `ProcessSingle`.

```csharp
var doc = new Document("Text to process", Language.English);
nlp.ProcessSingle(doc);
```

### Multiple Documents
For processing large numbers of documents, use `Process`. This method leverages multi-threading and lazy evaluation for better performance.

```csharp
IEnumerable<IDocument> docs = GetDocuments();
var processedDocs = nlp.Process(docs);

foreach(var doc in processedDocs)
{
    // Do something with the processed document
}
```

## Storing and Loading Pipelines

You can store a configured pipeline and its models into a single binary file and load it back later.

```csharp
// Store
using(var f = File.OpenWrite("my-pipeline.bin"))
{
    nlp.PackTo(f);
}

// Load
using(var f = File.OpenRead("my-pipeline.bin"))
{
    var nlp2 = await Pipeline.LoadFromPackedAsync(f);
}
```

## Neuralyzers
Neuralyzers are special components that can be added to the pipeline to correct mistakes made by other models (e.g., adding or forgetting entities) based on patterns.

```csharp
var neuralyzer = new Neuralyzer(Language.English, 0, "fixes");
neuralyzer.TeachAddPattern("Organization", "Amazon", mp => mp.Add(new PatternUnit(P.Single().WithToken("Amazon"))));
nlp.UseNeuralyzer(neuralyzer);
```
