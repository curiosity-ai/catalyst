using System;
using System.IO;
using System.Text;
using Xunit;

namespace Catalyst.Tests.DateTimeRecognition
{
    /// <summary>
    /// Prints what both engines make of a handful of inputs, side by side. Not an assertion — a magnifying
    /// glass for whichever construct is being worked on. Inputs come from CATALYST_DT_DEBUG, one per line.
    /// </summary>
    public class DebugTests
    {
        [Fact]
        public void Compare()
        {
            var raw = Environment.GetEnvironmentVariable("CATALYST_DT_DEBUG");
            if (string.IsNullOrWhiteSpace(raw)) return;

            var language  = Environment.GetEnvironmentVariable("CATALYST_DT_LANG") ?? "English";
            var inputs    = raw.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            var reference = DateTime.TryParse(Environment.GetEnvironmentVariable("CATALYST_DT_REF"), out var r) ? r : new DateTime(2016, 11, 7);
            var sb        = new StringBuilder();

            foreach (var line in inputs)
            {
                var input = line.TrimEnd('\r');
                sb.AppendLine($"INPUT  \"{input}\"   (ref {reference:yyyy-MM-dd HH:mm})");

                foreach (var hit in Engines.RunMicrosoft(language, input, reference)) sb.AppendLine("   MS  " + hit.Describe());
                foreach (var hit in Engines.RunCatalyst(language, input, reference))  sb.AppendLine("   CA  " + hit.Describe());

                sb.AppendLine();
            }

            var path = Path.Combine(AppContext.BaseDirectory, "parity", "debug.txt");
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, sb.ToString());
        }
    }
}
