using Mosaik.Core;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Catalyst
{
    public static partial class Spacy
    {
        public sealed class Pipeline : IDisposable
        {
            private dynamic _nlp;

            public Language Language { get; }
            public ModelSize ModelSize { get; }

            internal Pipeline(Language language, ModelSize modelSize, dynamic pipeline)
            {
                _nlp = pipeline;
                Language = language;
                ModelSize = modelSize;
            }

            public void ProcessSingle(Document document)
            {
                using (Py.GIL())
                {
                    var s_doc = _nlp(document.Value);
                    SyncBack(s_doc, document);
                }
            }

            public IEnumerable<Document> Process(IEnumerable<Document> documents)
            {
                var batch = new List<Document>();

                foreach(var doc in documents)
                {
                    batch.Add(doc);

                    if(batch.Count > 1000)
                    {
                        ProcessBatch(batch);
                        foreach(var processed in batch)
                        {
                            yield return processed;
                        }
                        batch.Clear();
                    }
                }

                ProcessBatch(batch);

                foreach (var processed in batch)
                {
                    yield return processed;
                }

                void ProcessBatch(List<Document> docs)
                {
                    if (docs.Count > 0)
                    {
                        using (Py.GIL())
                        {
                            var s_docs = _nlp.pipe(docs.Select(d => d.Value).ToArray());

                            for (int i = 0; i < docs.Count; i++)
                            {
                                SyncBack(s_docs[i], docs[i]);
                            }
                        }
                    }
                }
            }

            private void SyncBack(dynamic s_doc, Document document)
            {
                foreach (var s_sentence in s_doc.sents)
                {
                    var span = document.AddSpan((int)s_sentence.start_char, (int)s_sentence.end_char);
                    foreach (var s_token in s_sentence)
                    {
                        var tb = (int)s_token.idx;
                        var token = span.AddToken(tb, tb + (int)s_token.__len__() - 1);

                        token.POS = ConvertPOS((string)s_token.pos_);
                        token.DependencyType = (string)s_token.dep_;

                        var head = s_token.head;
                        if (head is object)
                        {
                            token.Head = (int)head.i;
                        }
                        else
                        {
                            token.Head = -1;
                        }
                    }
                }

                //for ent in doc.ents: print(ent.text, ent.start_char, ent.end_char, ent.label_)
            }

            private PartOfSpeech ConvertPOS(string s_pos)
            {
                return Enum.TryParse<PartOfSpeech>(s_pos, out var pos) ? pos : PartOfSpeech.X;
            }

            public void Dispose()
            {
                using (Py.GIL())
                {
                    _nlp = null;
                }
            }
        }
    }
}
