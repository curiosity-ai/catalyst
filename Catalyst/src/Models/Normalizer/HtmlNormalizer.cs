using HtmlAgilityPack;
using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UID;

namespace Catalyst.Models
{
    public class HtmlNormalizer : INormalizer, IProcess
    {
        public Language Language => Language.Any;
        public string Type => typeof(HtmlNormalizer).FullName;

        public string Tag => "";
        public int Version => 0;

        public static async Task<HtmlNormalizer> FromStoreAsync(Language language, int version, string tag)
        {
            return await Task.FromResult(new HtmlNormalizer());
        }

        public static async Task<bool> ExistsAsync(Language language, int version, string tag)
        {
            return true;
        } // Needs to say it exists, otherwise when calling ModelDescription.ExistsAsync(Language language, int version, string tag), it will fail to load this model

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            Normalize(document);
        }

        public void Normalize(IDocument document)
        {
            if(NeedNormalize(document.Value))
            {
                document.Value = GetTextFromHtml(document.Value);
            }
        }

		public string Normalize(string text)
		{
			if (NeedNormalize(text))
			{
				return GetTextFromHtml(text);
			}
			return text;
		}

		private static bool NeedNormalize(string content)
        {
			//Super simple heuristic to detect if text contains any <> html tags fast
            var s = content.AsSpan();

            var open = s.IndexOf('<');
            if(open >= 0)
            {
                var close = s.Slice(open).IndexOf('>');
                if (close >= 1) return true;
            }

			if(s.IndexOf("&nbsp;".AsSpan()) >= 0)
			{
				return true;
			}

            return false;
        }

		private static string GetTextFromHtml(string html)
		{
			if (string.IsNullOrEmpty(html)) return "";
			
			var htmlDoc = new HtmlDocument();
			
			htmlDoc.LoadHtml(html);

			var sb = StringExtensions.StringBuilderPool.Rent();

			GetTextFromNodes(sb, htmlDoc.DocumentNode.ChildNodes);

			var result = HtmlEntity.DeEntitize(sb.ToString());

			StringExtensions.StringBuilderPool.Return(sb);

			return result;
		}

		private static readonly HashSet<string> LineBreaks = new HashSet<string>(new[] { "p", "br", "table", "th", "tr" }, new LowerCaseComparer());
		private static readonly HashSet<string> IndentTags = new HashSet<string>(new []{ "ul", "li" }, new LowerCaseComparer());

		private static void GetTextFromNodes(StringBuilder sb, HtmlNodeCollection nodes, int indent = 0)
		{
			foreach (var node in nodes)
			{
				if (string.Equals(node.Name, "style", StringComparison.InvariantCultureIgnoreCase) 
		         || string.Equals(node.Name, "script", StringComparison.InvariantCultureIgnoreCase))
				{
					continue;
				}

				if (node.HasChildNodes)
				{
					if (IndentTags.Contains(node.Name))
					{
						GetTextFromNodes(sb, node.ChildNodes, indent + 1);
					}
					else
					{
						GetTextFromNodes(sb, node.ChildNodes, indent);
					}
				}
				else
				{
					var innerText = node.InnerText;
					if (!string.IsNullOrWhiteSpace(innerText))
					{
						AppendIndent();
						sb.Append(node.InnerText);
						if(sb.Length > 0 && sb[sb.Length - 1] != ' ')
						{
							sb.Append(' ');
						}
					}
				}

				if (LineBreaks.Contains(node.Name) && sb.Length > 0 && sb[sb.Length-1] != '\n')
				{
					sb.AppendLine();
				}
			}

			void AppendIndent()
			{
				for (int i = 0; i < indent; i++) sb.Append(' ');
			}
		}


		private class LowerCaseComparer : IEqualityComparer<string>
		{
			public bool Equals(string x, string y)
			{
				return string.Equals(x, y, StringComparison.InvariantCultureIgnoreCase);
			}

			public int GetHashCode(string obj)
			{
				return (obj ?? "").IgnoreCaseHash32();
			}
		}
	}
}