using System;
using System.Collections.ObjectModel;
using System.Reflection;

namespace Catalyst.Models.LDA
{
    /// <summary>
    /// Contains extension methods that aid in building cross platform.
    /// </summary>
    internal static class PlatformUtils
    {
        public static ReadOnlyCollection<T> AsReadOnly<T>(this T[] items)
        {
            if (items == null)
                return null;
            return new ReadOnlyCollection<T>(items);
        }

        public static bool IsGenericEx(this Type type, Type typeDef)
        {
            Contracts.AssertValue(type);
            Contracts.AssertValue(typeDef);
            var info = type.GetTypeInfo();
            return info.IsGenericType && info.GetGenericTypeDefinition() == typeDef;
        }

        public static Type[] GetGenericTypeArgumentsEx(this Type type)
        {
            Contracts.AssertValue(type);
            var typeInfo = IntrospectionExtensions.GetTypeInfo(type);
            return typeInfo.IsGenericTypeDefinition
                                ? typeInfo.GenericTypeParameters
                                : typeInfo.GenericTypeArguments;
        }
    }
}
