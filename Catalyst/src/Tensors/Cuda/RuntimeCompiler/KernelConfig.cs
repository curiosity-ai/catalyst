using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.RuntimeCompiler
{
    public class KernelConfig
    {
        private readonly SortedDictionary<string, string> values = new SortedDictionary<string, string>();


        public KernelConfig()
        {
        }

        public IEnumerable<string> Keys { get { return values.Keys; } }

        public IEnumerable<KeyValuePair<string, string>> AllValues()
        {
            return values;
        }

        public override bool Equals(object obj)
        {
            var o = obj as KernelConfig;
            if (o == null) return false;

            if (values.Count != o.values.Count) return false;

            foreach (var kvp in values)
            {
                string oValue;
                if (values.TryGetValue(kvp.Key, out oValue))
                {
                    if (!kvp.Value.Equals(oValue))
                        return false;
                }
                else
                {
                    return false;
                }
            }

            return true;
        }

        public override int GetHashCode()
        {
            int result = 0;
            foreach (var kvp in values)
            {
                result ^= kvp.Key.GetHashCode();
                result ^= kvp.Value.GetHashCode();
            }
            return result;
        }

        public bool ContainsKey(string name)
        {
            return values.ContainsKey(name);
        }

        public void Set(string name, string value)
        {
            values[name] = value;
        }

        public string ApplyToTemplate(string templateCode)
        {
            var fullCode = new StringBuilder();
            foreach (var item in values)
            {
                fullCode.AppendFormat("#define {0} {1}\n", item.Key, item.Value);
            }
            fullCode.AppendLine(templateCode);
            return fullCode.ToString();
        }
    }
}
