using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Catalyst;

namespace Catalyst.Presidio
{
    /// <summary>
    /// Implements privacy-preserving cloaking of PII entities.
    /// Replaces sensitive data with consistent pseudonyms (e.g., EMAIL_1) before sending to an external service,
    /// and then rehydrates the response to restore the original entities.
    /// </summary>
    /// <remarks>Not thread safe</remarks>
    public class PresidioCloaker
    {
        private readonly Dictionary<string, string> _entityToToken = new Dictionary<string, string>();
        private readonly Dictionary<string, string> _tokenToEntity = new Dictionary<string, string>();
        private readonly Dictionary<string, int> _typeCounts = new Dictionary<string, int>();

        // Maximum length of any generated token. Used to size the rolling buffer in the streaming version.
        private int _maxTokenLength = 0;

        /// <summary>
        /// Gets a consistent pseudonym for the given entity text and type.
        /// </summary>
        private string GetOrCreateToken(string entityText, string entityType)
        {
            if (_entityToToken.TryGetValue(entityText, out var existingToken))
            {
                return existingToken;
            }

            if (!_typeCounts.TryGetValue(entityType, out int count))
            {
                count = 0;
            }
            count++;
            _typeCounts[entityType] = count;

            string newToken = $"{entityType.ToUpperInvariant()}_{count}";
            _entityToToken[entityText] = newToken;
            _tokenToEntity[newToken] = entityText;

            if (newToken.Length > _maxTokenLength)
            {
                _maxTokenLength = newToken.Length;
            }

            return newToken;
        }

        private List<RecognizerResult> RemoveOverlaps(IEnumerable<RecognizerResult> results)
        {
            var sorted = results.OrderBy(r => r.Start).ThenByDescending(r => r.Score).ThenByDescending(r => r.Length).ToList();
            var output = new List<RecognizerResult>();

            int lastEnd = -1;

            foreach (var result in sorted)
            {
                if (result.Start >= lastEnd) // changed from > to >= to allow adjacent entities
                {
                    output.Add(result);
                    lastEnd = result.End;
                }
            }
            return output;
        }

        /// <summary>
        /// Replaces matching entities with cloaked versions, calls the input method, and replaces back in the output the cloaked entities with their original values.
        /// </summary>
        public async Task<string> CloakAsync(IDocument document, IEnumerable<RecognizerResult> results, Func<string, Task<string>> action)
        {
            if (document == null) throw new ArgumentNullException(nameof(document));
            if (action == null) throw new ArgumentNullException(nameof(action));

            string text = document.Value;
            if (string.IsNullOrEmpty(text))
            {
                return await action(text).ConfigureAwait(false);
            }

            if (results == null || !results.Any())
            {
                return await action(text).ConfigureAwait(false);
            }

            var filteredResults = RemoveOverlaps(results);
            var sortedResults = filteredResults.OrderByDescending(r => r.Start).ToList();
            var sb = new StringBuilder(text);

            foreach (var result in sortedResults)
            {
                if (result.Start < 0 || result.End > text.Length || result.Start > result.End) continue;

                string originalText = text.Substring(result.Start, result.Length);
                string token = GetOrCreateToken(originalText, result.EntityType);

                sb.Remove(result.Start, result.Length);
                sb.Insert(result.Start, token);
            }

            string cloakedInput = sb.ToString();

            // Call the external service
            string response = await action(cloakedInput).ConfigureAwait(false);

            if (string.IsNullOrEmpty(response))
            {
                return response;
            }

            // Rehydrate the response
            return Rehydrate(response);
        }

        /// <summary>
        /// Replaces matching entities with cloaked versions, calls the input method, and replaces back in the output the cloaked entities with their original values in a streaming fashion.
        /// </summary>
        public async IAsyncEnumerable<string> CloakAsync(IDocument document, IEnumerable<RecognizerResult> results, Func<string, IAsyncEnumerable<string>> action)
        {
            if (document == null) throw new ArgumentNullException(nameof(document));
            if (action == null) throw new ArgumentNullException(nameof(action));

            string text = document.Value;
            if (string.IsNullOrEmpty(text) || results == null || !results.Any())
            {
                await foreach (var chunk in action(text).ConfigureAwait(false))
                {
                    yield return chunk;
                }
                yield break;
            }

            var filteredResults = RemoveOverlaps(results);
            var sortedResults = filteredResults.OrderByDescending(r => r.Start).ToList();
            var sb = new StringBuilder(text);

            foreach (var result in sortedResults)
            {
                if (result.Start < 0 || result.End >= text.Length || result.Start > result.End) continue;

                string originalText = text.Substring(result.Start, result.Length);
                string token = GetOrCreateToken(originalText, result.EntityType);

                sb.Remove(result.Start, result.Length);
                sb.Insert(result.Start, token);
            }

            string cloakedInput = sb.ToString();

            // Stream rehydration
            var buffer = new StringBuilder();

            int currentMaxTokenLength = _maxTokenLength;

            // To ensure we don't miss tokens split across chunks, we need to hold back at least MaxTokenLength characters,
            // or perhaps more precisely, hold back enough to ensure we don't truncate a potential token.

            await foreach (var chunk in action(cloakedInput).ConfigureAwait(false))
            {
                if (string.IsNullOrEmpty(chunk)) continue;

                buffer.Append(chunk);

                while (buffer.Length >= currentMaxTokenLength * 2) // Maintain a buffer to look for tokens
                {
                    // Process part of the buffer
                    // Find the safest point to yield. A safe point is after we check for tokens.
                    // Let's replace tokens in the entire buffer, then yield everything except the last MaxTokenLength characters.

                    RehydrateBuffer(buffer);

                    if (buffer.Length > currentMaxTokenLength)
                    {
                        int yieldLen = buffer.Length - currentMaxTokenLength;
                        string toYield = buffer.ToString(0, yieldLen);
                        buffer.Remove(0, yieldLen);
                        yield return toYield;
                    }
                }
            }

            // Flush the remaining buffer
            if (buffer.Length > 0)
            {
                RehydrateBuffer(buffer);
                yield return buffer.ToString();
            }
        }

        private string Rehydrate(string text)
        {
            var sb = new StringBuilder(text);
            RehydrateBuffer(sb);
            return sb.ToString();
        }

        private void RehydrateBuffer(StringBuilder sb)
        {
            // Simple replace of all known tokens.
            // Since tokens like ORG_1 could be prefixes of ORG_10, we sort tokens by length descending to replace ORG_10 before ORG_1.

            foreach (var kvp in _tokenToEntity.OrderByDescending(kv => kv.Key.Length))
            {
                string token = kvp.Key;
                string originalEntity = kvp.Value;

                sb.Replace(token, originalEntity);
            }
        }
    }
}