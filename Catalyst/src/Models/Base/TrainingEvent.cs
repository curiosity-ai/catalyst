using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Models
{
    public class TrainingEvent : EventArgs
    {
        public float Loss { get; set; }
        public int Epoch { get; set; }
        public int Update { get; set; }
        public int DocumentCount { get; set; }
        public int SpansCount { get; set; }
        public long TokensCount { get; set; }
        public DateTimeOffset StartDateTime { get; set; }
        public float LearningRate { get; set; }
    }
}
