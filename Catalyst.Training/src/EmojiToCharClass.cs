using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeoSmart.Unicode;
namespace Catalyst.Training
{
    public static class EmojiToCharClass
    {
        //This class is used to create the fast emoji detection method CharSpanExtensions.IsEmoji - only needed if we need to update the method
        public static void PrintEmojiTestClass()
        {
            Console.OutputEncoding = new UTF8Encoding();
            Console.InputEncoding  = new UTF8Encoding();

            var p = new ConcurrentDictionary<int, HashSet<ushort>>();


            int k = 0;
            foreach (var emoji in Emoji.All)
            {
                var chars = emoji.Sequence.AsUtf16();
                //Console.WriteLine(emoji.ToString() + "\t" + string.Join("\t", chars.Select(c => c.ToString())));

                k = 0;
                foreach (var c in chars)
                {
                    var d = p.GetOrAdd(k, new HashSet<ushort>());
                    k++;
                    d.Add(c);
                }
            }


            k = p.Count-1;
            var fullTest = "";

            foreach(var kv in p.OrderByDescending(kv2 => kv2.Key))
            {

                var values = kv.Value;
                var v = (char)('a' + k);
                var test = string.Join(" || ", values.Select(c => $"({v} == {c})"));


                fullTest = @"
if(candidate.Length > " + (k) + @")
{
var " + v + @" = (ushort)candidate[" + k + @"];
if(" + test + @")
{
" + fullTest + @"
" + ((k % 2 == 1)? @"count = " + (k+1) + @";return true;" : "" ) + @"
}
}";

                k--;
            }

            Console.WriteLine(fullTest);
            Console.ReadLine();
        }
    }
}
