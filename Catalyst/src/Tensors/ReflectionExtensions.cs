using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;

namespace Catalyst.Tensors
{
    public static class AssemblyExtensions
    {
        public static IEnumerable<Tuple<Type, IEnumerable<T>>> TypesWithAttribute<T>(this Assembly assembly, bool inherit)
        {
            foreach (var type in assembly.GetTypes())
            {
                var attributes = type.GetCustomAttributes(typeof(T), inherit);
                if (attributes.Any())
                {
                    yield return Tuple.Create(type, attributes.Cast<T>());
                }
            }
        }
    }

    public static class TypeExtensions
    {
        public static IEnumerable<Tuple<MethodInfo, IEnumerable<T>>> MethodsWithAttribute<T>(this Type type, bool inherit)
        {
            foreach (var method in type.GetMethods())
            {
                var attributes = method.GetCustomAttributes(typeof(T), inherit);
                if (attributes.Any())
                {
                    yield return Tuple.Create(method, attributes.Cast<T>());
                }
            }
        }
    }

    public static class MethodExtensions
    {
        public static IEnumerable<Tuple<ParameterInfo, IEnumerable<T>>> ParametersWithAttribute<T>(this MethodInfo method, bool inherit)
        {
            foreach (var paramter in method.GetParameters())
            {
                var attributes = paramter.GetCustomAttributes(typeof(T), inherit);
                if (attributes.Any())
                {
                    yield return Tuple.Create(paramter, attributes.Cast<T>());
                }
            }
        }
    }
}
