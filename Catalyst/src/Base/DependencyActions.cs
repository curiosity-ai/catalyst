// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

namespace Catalyst
{
    //public static class DependencyActions
    //{
    //    public const string LeftArc  = "LeftArc";
    //    public const string RightArc = "RightArc";
    //    public const string Reduce   = "Reduce";
    //    public const string Shift    = "Shift";
    //    public const string Root     = "Root";

    //    public static string[] ConstructListOfActions(IList<string> arcNames)
    //    {
    //        var list = new List<string>() { Shift, LeftArc, RightArc};
    //        //foreach (var an in arcNames)
    //        //{
    //        //    list.Add(GetMoveName(LeftArc,  an));
    //        //    list.Add(GetMoveName(RightArc, an));
    //        //}

    //        return list.ToArray();
    //    }

    //    public static string GetMoveName (string action, string type)
    //    {
    //        if(string.IsNullOrEmpty(type)) { return action; }
    //        return $"{action}_{type}";
    //    }

    //    public static string GetLabel (string action)
    //    {
    //        int i = action.IndexOf('_');
    //        return i < 0 ? "" : action.Substring(i+1);
    //    }

    //    public const int SHIFT  = 0;
    //    public const int RIGHT = 1;
    //    public const int LEFT   = 2;

    //    public static int GetMove(string action)
    //    {
    //        //Do not change these index, as they reflect on the AveragePerceptronDependencyParser.cs file
    //        if (action.StartsWith(LeftArc))  { return LEFT; }
    //        if (action.StartsWith(RightArc)) { return RIGHT; }
    //        if (action.StartsWith(Shift))    { return SHIFT; }
    //        throw new InvalidOperationException("Invalid action name:" + action);
    //    }
    //}
}