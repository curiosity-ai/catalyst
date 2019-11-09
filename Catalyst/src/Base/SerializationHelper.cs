using System;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace Catalyst
{
    internal static class SerializationHelper
    {
        internal static Func<byte[], object> CreateDeserializer(Type inputType)
        {
            var genericDeserializer = typeof(MessagePack.LZ4MessagePackSerializer).GetMethods().Where(m => m.Name == "Deserialize" && m.GetParameters().Count() == 1 && m.GetParameters().First().Name == "bytes" && m.GetParameters().First().ParameterType == typeof(Byte[])).Single();
            var constructedDeserializer = genericDeserializer.MakeGenericMethod(new Type[] { inputType });

            var genericHelper = typeof(SerializationHelper).GetMethod(nameof(CreateHelper), BindingFlags.Static | BindingFlags.NonPublic);
            var constructedHelper = genericHelper.MakeGenericMethod(typeof(byte[]), constructedDeserializer.ReturnType);

            var genericConverter = typeof(SerializationHelper).GetMethod(nameof(ConvertFunc), BindingFlags.Static | BindingFlags.NonPublic);
            var constructedConverter = genericConverter.MakeGenericMethod(typeof(byte[]), constructedDeserializer.ReturnType);

            var ret = constructedHelper.Invoke(null, new object[] { constructedDeserializer });
            return constructedConverter.Invoke(null, new object[] { ret, typeof(byte[]), typeof(object) }) as Func<byte[], object>;
        }

        private static Func<TTarget, TReturn> CreateHelper<TTarget, TReturn>(MethodInfo method) where TTarget : class
        {
            Func<TTarget, TReturn> func = (Func<TTarget, TReturn>)Delegate.CreateDelegate(typeof(Func<TTarget, TReturn>), method);
            Func<TTarget, TReturn> ret = (TTarget target) => func(target);
            return ret;
        }

        private static Delegate ConvertFunc<TInIn, TInOut>(Func<TInIn, TInOut> func, Type argType, Type resultType)
        {
            var param = Expression.Parameter(argType);
            var convertedParam = new Expression[] { Expression.Convert(param, typeof(TInIn)) };

            // This is gnarly... If a func contains a closure, then even though its static, its first
            // param is used to carry the closure, so its as if it is not a static method, so we need
            // to check for that param and call the func with it if it has one...
            Expression call;
            call = Expression.Convert(func.Target == null ? Expression.Call(func.Method, convertedParam)
                                                          : Expression.Call(Expression.Constant(func.Target), func.Method, convertedParam), resultType);

            var delegateType = typeof(Func<,>).MakeGenericType(argType, resultType);
            return Expression.Lambda(delegateType, call, param).Compile();
        }

    }
}