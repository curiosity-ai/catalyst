namespace Catalyst.Models
{
    public class BeamAction
    {
        public float DeltaScore;
        public int Index;

        public BeamAction(int index, float delta)
        {
            Index = index; DeltaScore = delta;
        }
    }
}